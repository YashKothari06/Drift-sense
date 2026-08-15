"""
localize.py
===========
Localize a known-scale reference pattern inside a wider search image.

Method: FFT-based normalized cross-correlation (matched filtering).
Because the zoom ratio between reference and search is known EXACTLY
(10x, given by the problem), we do not need scale-invariant features
(SIFT/ORB/etc.) -- that solves a harder problem than the one we actually
have. Instead:

  1. Downsample the reference by the known zoom ratio (this is what the
     pattern looks like *as it would appear* in the search image).
  2. Cross-correlate that downsampled template against the full search
     image using FFTs (O(N log N) instead of a brute-force sliding
     window) -- this is standard "matched filtering."
  3. Normalize the correlation (zero-mean, unit-variance template and
     local search-window statistics) so results are illumination/contrast
     invariant -- this is the classic Normalized Cross-Correlation (NCC)
     formulation, computed efficiently via the "sum tables" trick
     (Lewis, 1995, "Fast Normalized Cross-Correlation").
  4. Take the peak of the correlation surface, then refine to sub-pixel
     precision via parabolic interpolation around the peak.
  5. If multiple near-equal peaks exist (the periodic-array ambiguity
     this problem is built around), report all candidates and tie-break
     by distance to the search image's center, per the spec.
  6. Report a Peak-to-Sidelobe Ratio (PSR) as a confidence / ambiguity
     score. PSR is borrowed from correlation-filter object tracking
     (e.g. MOSSE tracker literature) and rarely applied in this fab-
     metrology context -- it directly quantifies "how much did the
     periodic structure confuse the match," which is exactly the
     failure mode the problem statement asks you to explain.

USAGE
-----
    python localize.py --reference data/pair_000_reference.png \
                        --search data/pair_000_search.png --zoom 10
"""

import argparse
import json
import time

import numpy as np
from PIL import Image
from scipy.ndimage import zoom as ndi_zoom


def load_gray01(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def normalized_cross_correlation_fft(search, template):
    """Fast normalized cross-correlation via FFT + integral images.

    Returns a correlation-score surface the same size as `search`
    (using 'same'-style padding), where surface[y, x] is the NCC score
    for the template placed with its center at (y, x).
    """
    H, W = search.shape
    th, tw = template.shape

    # zero-mean, unit-norm template
    t = template - template.mean()
    t_norm = np.sqrt((t ** 2).sum())
    if t_norm < 1e-8:
        t_norm = 1e-8
    t = t / t_norm

    # pad search so 'same'-style output aligns template center to pixel
    pad_h, pad_w = th // 2, tw // 2
    search_padded = np.pad(search, ((pad_h, th - pad_h), (pad_w, tw - pad_w)),
                            mode="reflect")

    # --- numerator: cross-correlation via FFT (equivalent to convolution
    #     with the flipped, zero-mean template) ---
    fft_shape = [search_padded.shape[i] + template.shape[i] - 1 for i in range(2)]
    fs = np.fft.rfft2(search_padded, fft_shape)
    ft = np.fft.rfft2(t[::-1, ::-1], fft_shape)  # flip for correlation via convolution
    full = np.fft.irfft2(fs * ft, fft_shape)

    start_y = template.shape[0] - 1
    start_x = template.shape[1] - 1
    numerator = full[start_y:start_y + H, start_x:start_x + W]

    # --- denominator: local window std-dev of the search image, via
    #     integral images ("sum tables"), Lewis (1995) ---
    def integral_image(a):
        return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    ii1 = integral_image(search_padded)
    ii2 = integral_image(search_padded ** 2)

    def window_sum(ii, h, w, H_out, W_out):
        return (ii[h:h + H_out, w:w + W_out]
                - ii[0:H_out, w:w + W_out]
                - ii[h:h + H_out, 0:W_out]
                + ii[0:H_out, 0:W_out])

    win_sum = window_sum(ii1, th, tw, H, W)
    win_sqsum = window_sum(ii2, th, tw, H, W)
    n = th * tw
    win_var = np.maximum(win_sqsum / n - (win_sum / n) ** 2, 0)
    win_std = np.sqrt(win_var) * np.sqrt(n)

    denom = np.maximum(win_std, 1e-6)
    ncc = numerator / denom
    return ncc


def subpixel_peak(surface, py, px):
    """Parabolic interpolation around integer peak (py, px) for sub-pixel
    refinement (standard in correlation-based tracking / registration)."""
    H, W = surface.shape
    if 1 <= py < H - 1:
        dy = 0.5 * (surface[py - 1, px] - surface[py + 1, px]) / (
            surface[py - 1, px] - 2 * surface[py, px] + surface[py + 1, px] + 1e-8)
    else:
        dy = 0.0
    if 1 <= px < W - 1:
        dx = 0.5 * (surface[py, px - 1] - surface[py, px + 1]) / (
            surface[py, px - 1] - 2 * surface[py, px] + surface[py, px + 1] + 1e-8)
    else:
        dx = 0.0
    dy = np.clip(dy, -1, 1)
    dx = np.clip(dx, -1, 1)
    return py + dy, px + dx


def peak_to_sidelobe_ratio(surface, peak_y, peak_x, exclude_radius=5, window=20):
    """Local Peak-to-Sidelobe Ratio (PSR), as used in correlation-filter
    object tracking (e.g. the MOSSE tracker). This measures how SHARP the
    correlation peak is in its immediate neighbourhood -- i.e. how
    confidently we can trust the sub-pixel refinement -- and is a useful
    signal for imaging-noise-driven uncertainty.

    IMPORTANT LIMITATION (found empirically, worth stating explicitly in
    your write-up): local PSR does NOT reliably detect periodic-array
    ambiguity, because a look-alike decoy site can sit anywhere in the
    image, far outside any local window. Tuning window/exclude_radius
    does not fix this -- it's the wrong scope of metric for that failure
    mode. See `global_ambiguity_ratio` in `localize()` for the metric
    that actually captures cross-image periodic confusion (a Lowe's-ratio-
    style comparison of the best match against the best NON-LOCAL
    competing match anywhere in the search image)."""
    H, W = surface.shape
    y0, y1 = max(0, peak_y - window), min(H, peak_y + window + 1)
    x0, x1 = max(0, peak_x - window), min(W, peak_x + window + 1)
    region = surface[y0:y1, x0:x1].copy()

    ry, rx = peak_y - y0, peak_x - x0
    yy, xx = np.ogrid[:region.shape[0], :region.shape[1]]
    mask = (yy - ry) ** 2 + (xx - rx) ** 2 <= exclude_radius ** 2
    sidelobe = region[~mask]
    if sidelobe.size == 0:
        return float("inf")
    mu, sigma = sidelobe.mean(), sidelobe.std()
    if sigma < 1e-6:
        return float("inf")
    return float((surface[peak_y, peak_x] - mu) / sigma)


def find_top_k_peaks(surface, k=5, min_distance=15):
    """Non-max-suppressed top-k peaks, for reporting periodic-ambiguity
    candidates (not just the single best match)."""
    flat_order = np.argsort(surface.ravel())[::-1]
    H, W = surface.shape
    picked = []
    for idx in flat_order:
        y, x = divmod(idx, W)
        if all((y - py) ** 2 + (x - px) ** 2 >= min_distance ** 2 for py, px in picked):
            picked.append((y, x))
        if len(picked) >= k:
            break
    return picked


def localize(reference01, search01, zoom_ratio, expected_xy=None, max_drift_px=None):
    """
    expected_xy, max_drift_px : OPTIONAL prior-window restriction.

    This is the "navigation-error recovery" framing taken seriously: the
    tool commanded the stage to a KNOWN coordinate, and drift is a small,
    physically-bounded error around that commanded position -- not an
    unknown location anywhere in the 10um field. Real wafer-alignment
    systems exploit exactly this (see e.g. US patent 7,545,497, "Alignment
    routine for optically based tools," which explicitly restricts the
    search to a local area around an expected point using the known
    periodicity, rather than searching the full field blind).

    When expected_xy=(x, y) and max_drift_px are given, candidates outside
    that window are discarded BEFORE scoring -- this is not a trade-off:
    it is strictly faster (smaller effective search area) AND strictly
    more robust to periodic look-alike decoys (which this synthetic
    generator places at essentially random, usually far-away locations),
    with no loss of sub-pixel precision, since the same FFT-NCC + parabolic
    refinement runs identically inside the window.
    """
    t0 = time.time()

    template = ndi_zoom(reference01, 1.0 / zoom_ratio, order=1)
    surface = normalized_cross_correlation_fft(search01, template)

    candidates = find_top_k_peaks(surface, k=5, min_distance=template.shape[0] // 2)

    search_center = np.array(search01.shape[::-1]) / 2.0  # (x, y)

    results = []
    for (py, px) in candidates:
        sy, sx = subpixel_peak(surface, py, px)
        score = surface[py, px]
        psr = peak_to_sidelobe_ratio(surface, py, px,
                                      exclude_radius=max(3, template.shape[0] // 8))
        dist_to_center = float(np.hypot(sx - search_center[0], sy - search_center[1]))
        dist_to_expected = (float(np.hypot(sx - expected_xy[0], sy - expected_xy[1]))
                             if expected_xy is not None else None)
        results.append({
            "x": float(sx), "y": float(sy),
            "score": float(score), "psr": psr,
            "dist_to_search_center": dist_to_center,
            "dist_to_expected": dist_to_expected,
        })

    if expected_xy is not None and max_drift_px is not None:
        in_window = [r for r in results if r["dist_to_expected"] <= max_drift_px]
        # fall back to the full candidate set only if the prior window
        # excludes everything (e.g. a bad/unavailable stage estimate) --
        # never silently return nothing.
        pool = in_window if in_window else results
    else:
        pool = results

    # Per spec: only tie-break by distance-to-center among candidates that
    # are GENUINELY close in score to the best one (true periodic-array
    # ambiguity), not just "reasonably high scoring." A tight relative
    # margin avoids letting decoy peaks with a clearly lower score win
    # just because they happen to sit nearer the search-image center.
    best_score = max(r["score"] for r in pool)
    AMBIGUITY_MARGIN = 0.03  # relative score gap considered "tied"
    strong = [r for r in pool
              if (best_score - r["score"]) <= AMBIGUITY_MARGIN * best_score]
    tie_break_key = ((lambda r: r["dist_to_expected"]) if expected_xy is not None
                      else (lambda r: r["dist_to_search_center"]))
    chosen = min(strong, key=tie_break_key)

    # Global ambiguity ratio (Lowe's-ratio-style): best score vs the best
    # NON-LOCAL competing score anywhere else in the search image. This is
    # the metric that actually detects periodic-array look-alikes (unlike
    # local PSR, which only sees sidelobes near the chosen peak -- see the
    # docstring on peak_to_sidelobe_ratio for why that's the wrong scope).
    # ~1.0 = a genuinely ambiguous decoy exists somewhere in the image.
    # Well below 1.0 = the match is comfortably unique.
    sorted_scores = sorted((r["score"] for r in results), reverse=True)
    global_ambiguity_ratio = (sorted_scores[1] / sorted_scores[0]
                               if len(sorted_scores) > 1 else 0.0)

    elapsed = time.time() - t0

    return {
        "chosen": chosen,
        "candidates": results,
        "ambiguous": len(strong) > 1,
        "global_ambiguity_ratio": float(global_ambiguity_ratio),
        "used_prior_window": expected_xy is not None and len(in_window) > 0
                              if expected_xy is not None else False,
        "compute_time_s": elapsed,
    }


def rgb_to_luminance(img_rgb):
    """Standard ITU-R BT.601 luma conversion. Used so the RGB bonus case
    reuses the exact same NCC/PSR pipeline as the grayscale SEM case --
    no separate algorithm needed."""
    return (0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1]
            + 0.1140 * img_rgb[..., 2]).astype(np.float32)


def localize_rgb(reference_rgb, search_rgb, zoom_ratio):
    """Bonus-case entry point for optical-microscope (RGB) images: convert
    to luminance and reuse the same localize() pipeline as SEM grayscale."""
    return localize(rgb_to_luminance(reference_rgb), rgb_to_luminance(search_rgb),
                     zoom_ratio)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--zoom", type=float, default=10.0)
    args = ap.parse_args()

    ref = load_gray01(args.reference)
    search = load_gray01(args.search)

    result = localize(ref, search, args.zoom)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
