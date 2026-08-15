"""
generate_dataset.py
====================
Synthetic DRAM-style SEM image-pair generator for the "Drift-Sense" problem
(Semicon India Hackathon / Applied Materials).

Produces (reference, search, ground_truth) triples where:
  - reference: 1000x1000 grayscale, 1 nm/pixel (100x zoom), 1um x 1um FOV.
               Represents a previously-characterized site on the wafer.
  - search:    1000x1000 grayscale, 10 nm/pixel (10x zoom), 10um x 10um FOV.
               Contains the reference pattern shrunk by exactly 10x, embedded
               at a random location, surrounded by visually-similar periodic
               repeat units (the "needle in a haystack" difficulty).
  - ground_truth: dict with the true (x, y) center of the embedded pattern
               in search-image pixel coordinates (float, sub-pixel).

DESIGN NOTES (why the pattern looks the way it does)
------------------------------------------------------
DRAM arrays are built from repeating word-line / bit-line / storage-node
contact unit cells. We render this as:
  - horizontal "word lines"  (thick bars, periodic in y)
  - vertical   "bit lines"   (thick bars, periodic in x)
  - contacts at line intersections (small discs)
Each line/contact edge gets *line-edge roughness* (LER): a smoothed random
perturbation of the edge position, rather than a perfectly straight edge.
This is because real lithographic/etch edges are stochastically rough, not
geometrically perfect -- this is well documented in CD-SEM metrology
literature on LER/LWR (line-edge/line-width roughness) characterization.
    -> When you write your citations slide, look up:
       - LER/LWR metrology papers (SPIE Journal of Micro/Nanolithography)
       - SEM image noise models (Poisson-Gaussian mixed noise, e.g. work on
         "shot noise" in electron-beam imaging)
       - DRAM cell array pitch/CD scaling papers or textbooks (for realistic
         pitch/CD numbers to justify your parameter choices)

Because the *same* nominal LER-generation process is used independently for
every repeat unit in the search image, neighbouring "identical" DRAM cells
end up visually similar but not pixel-identical -- exactly mirroring how
real periodic structures create ambiguous look-alikes for a localization
algorithm, without making the problem artificially easy or impossible.

Noise added AFTER pattern rendering, independently to reference and search,
models the imaging pipeline itself:
  - Poisson (shot) noise: electron-count-limited signal -> signal-dependent
    noise, the dominant term in real SEM images.
  - Gaussian read/amplifier noise: additive, signal-independent.
  - Beam-spot blur: Gaussian PSF blur.
  - Scan-line jitter: independent per-row intensity offset (raster scan
    instability).
  - Search image is deliberately noisier than reference, per the FAQ
    ("assume wide-search image will be more noisy in test data").

USAGE
-----
    python generate_dataset.py --out ./data --n 5 --seed 42

Produces PNGs + a ground_truth.json in the output directory.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


# ----------------------------------------------------------------------
# Pattern rendering
# ----------------------------------------------------------------------

def _smooth_noise_1d(n, rng, correlation_px=8.0, amplitude=1.5):
    """Low-frequency random signal used to jitter an edge position.

    A short correlation length keeps roughness locally random (like real
    LER) while still being spatially smooth pixel-to-pixel.
    """
    raw = rng.normal(0, 1, size=n)
    smoothed = gaussian_filter(raw, sigma=correlation_px, mode="wrap")
    smoothed /= (smoothed.std() + 1e-8)
    return smoothed * amplitude


def render_dram_pattern(
    size,
    pitch_px,
    line_width_px,
    contact_radius_px,
    rng,
    ler_amplitude_px=1.2,
    ler_correlation_px=6.0,
    defect_prob=0.03,
):
    """Render a DRAM-style word-line/bit-line/contact array.

    Parameters are all in *pixels at this image's own resolution*, so the
    same function is reused for both the 1 nm/px reference scale and the
    10 nm/px search scale by passing scaled-down pitch/width/radius values.

    Returns a float32 array in [0, 1], where higher = brighter (conductive
    line/contact material), lower = background/dielectric.
    """
    canvas = np.zeros((size, size), dtype=np.float32)
    row_idx = np.arange(size)[:, None]   # column vector
    col_idx = np.arange(size)[None, :]   # row vector

    n_h_lines = int(size // pitch_px) + 2
    n_v_lines = int(size // pitch_px) + 2

    # --- horizontal word lines ---
    for i in range(n_h_lines):
        center = i * pitch_px + pitch_px / 2 - pitch_px  # allow partial line at edge
        jitter = _smooth_noise_1d(size, rng, ler_correlation_px, ler_amplitude_px)
        top = center - line_width_px / 2 + jitter
        bottom = center + line_width_px / 2 + jitter
        mask = (row_idx >= top[None, :]) & (row_idx <= bottom[None, :])
        canvas[mask] = np.maximum(canvas[mask], 0.65)

    # --- vertical bit lines ---
    for j in range(n_v_lines):
        center = j * pitch_px + pitch_px / 2 - pitch_px
        jitter = _smooth_noise_1d(size, rng, ler_correlation_px, ler_amplitude_px)
        left = center - line_width_px / 2 + jitter
        right = center + line_width_px / 2 + jitter
        mask = (col_idx >= left[:, None]) & (col_idx <= right[:, None])
        canvas[mask] = np.maximum(canvas[mask], 0.55)

    # --- storage-node contacts at intersections ---
    # NOTE: distance is computed only in a small local window around each
    # contact (not the full canvas) -- at search-image scale there can be
    # ~10^4 contacts, and a full-image distance calc per contact would be
    # O(n_contacts * size^2), which is prohibitively slow.
    for i in range(n_h_lines):
        cy = i * pitch_px + pitch_px / 2 - pitch_px
        for j in range(n_v_lines):
            cx = j * pitch_px + pitch_px / 2 - pitch_px
            if not (-contact_radius_px <= cy <= size + contact_radius_px):
                continue
            if not (-contact_radius_px <= cx <= size + contact_radius_px):
                continue
            # small per-contact jitter + radius variation (process variation)
            dy = rng.normal(0, 0.8)
            dx = rng.normal(0, 0.8)
            r = contact_radius_px * (1.0 + rng.normal(0, 0.08))

            # random missing-contact / extra-particle defects (realistic
            # stochastic yield-loss / contamination events)
            if rng.random() < defect_prob:
                continue  # missing contact

            ccy, ccx = cy + dy, cx + dx
            pad = r + 1.5
            y0, y1 = max(0, int(ccy - pad)), min(size, int(ccy + pad) + 1)
            x0, x1 = max(0, int(ccx - pad)), min(size, int(ccx + pad) + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            yy = np.arange(y0, y1, dtype=np.float32)[:, None] - ccy
            xx = np.arange(x0, x1, dtype=np.float32)[None, :] - ccx
            disc = (yy ** 2 + xx ** 2) <= r ** 2
            canvas[y0:y1, x0:x1][disc] = 1.0

    # occasional stray particle contamination (bright blob, not part of grid)
    if rng.random() < defect_prob * 2:
        py = rng.uniform(0, size)
        px = rng.uniform(0, size)
        pr = rng.uniform(2, 6)
        yy = row_idx.astype(np.float32) - py
        xx = col_idx.astype(np.float32) - px
        blob = (yy ** 2 + xx ** 2) <= pr ** 2
        canvas[blob] = 1.0

    return np.clip(canvas, 0, 1)


def render_finfet_pattern(
    size,
    pitch_px,
    line_width_px,
    contact_radius_px,
    rng,
    ler_amplitude_px=1.2,
    ler_correlation_px=6.0,
    defect_prob=0.03,
):
    """Render a FinFET-style array: parallel vertical fins crossed by
    horizontal gate lines, each edge with line-edge roughness (LER/LWR),
    as documented in FinFET CD-SEM metrology literature. `contact_radius_px`
    is reused as the gate-line width scale for signature parity with
    render_dram_pattern (so both are drop-in interchangeable in
    generate_pair).
    """
    canvas = np.zeros((size, size), dtype=np.float32)
    row_idx = np.arange(size)[:, None]
    col_idx = np.arange(size)[None, :]

    fin_pitch = pitch_px * 0.6          # fins are typically finer-pitched than gates
    gate_pitch = pitch_px
    fin_width = line_width_px * 0.5
    gate_width = contact_radius_px * 2.2

    n_fins = int(size // fin_pitch) + 2
    n_gates = int(size // gate_pitch) + 2

    # --- parallel fins (vertical, dim -- these sit "under" the gates) ---
    for j in range(n_fins):
        center = j * fin_pitch + fin_pitch / 2 - fin_pitch
        jitter = _smooth_noise_1d(size, rng, ler_correlation_px, ler_amplitude_px * 0.6)
        left = center - fin_width / 2 + jitter
        right = center + fin_width / 2 + jitter
        mask = (col_idx >= left[:, None]) & (col_idx <= right[:, None])
        canvas[mask] = np.maximum(canvas[mask], 0.35)

    # --- gate lines (horizontal, bright -- poly/metal gate over fins) ---
    for i in range(n_gates):
        center = i * gate_pitch + gate_pitch / 2 - gate_pitch
        if rng.random() < defect_prob:
            continue  # missing/broken gate line (process defect)
        jitter = _smooth_noise_1d(size, rng, ler_correlation_px, ler_amplitude_px)
        top = center - gate_width / 2 + jitter
        bottom = center + gate_width / 2 + jitter
        mask = (row_idx >= top[None, :]) & (row_idx <= bottom[None, :])
        canvas[mask] = np.maximum(canvas[mask], 0.9)

    # occasional stray particle contamination
    if rng.random() < defect_prob * 2:
        py = rng.uniform(0, size)
        px = rng.uniform(0, size)
        pr = rng.uniform(2, 6)
        yy = row_idx.astype(np.float32) - py
        xx = col_idx.astype(np.float32) - px
        blob = (yy ** 2 + xx ** 2) <= pr ** 2
        canvas[blob] = 1.0

    return np.clip(canvas, 0, 1)


# ----------------------------------------------------------------------
# Imaging-pipeline noise (applied after clean pattern rendering)
# ----------------------------------------------------------------------

def apply_sem_noise(
    img01,
    rng,
    photon_gain=80.0,      # lower = noisier (fewer "electrons" per bright pixel)
    read_noise_std=0.01,
    blur_sigma=0.8,
    scanline_std=0.015,
):
    """Apply a mixed Poisson-Gaussian SEM-like noise model plus beam blur
    and scan-line jitter.

    img01 : float32 array in [0, 1]
    """
    img = gaussian_filter(img01, sigma=blur_sigma)  # beam-spot PSF blur

    # Poisson shot noise: treat pixel intensity as expected photon/electron
    # count scaled by photon_gain, sample, then rescale back to [0,1].
    lam = np.clip(img, 0, 1) * photon_gain
    shot = rng.poisson(lam).astype(np.float32) / photon_gain

    # Gaussian read/amplifier noise
    read = rng.normal(0, read_noise_std, size=img.shape).astype(np.float32)

    # per-row scan-line intensity jitter (raster instability)
    row_jitter = rng.normal(0, scanline_std, size=(img.shape[0], 1)).astype(np.float32)

    out = shot + read + row_jitter
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------
# Pair generation
# ----------------------------------------------------------------------

def generate_pair(size=1000, zoom_ratio=10, seed=None,
                   ref_pitch_nm=100.0, ref_line_width_nm=40.0, ref_contact_r_nm=15.0,
                   structure="dram", difficulty="easy"):
    """Generate one (reference, search, ground_truth) triple.

    ref_pitch_nm etc. are physical array parameters in nanometres. At the
    reference resolution (1 nm/px) these map 1:1 to pixels; at the search
    resolution (10 nm/px) they are divided by `zoom_ratio`.

    structure  : "dram" (word-line/bit-line/contact array) or
                 "finfet" (parallel fin + gate-line array).
    difficulty : "easy"   -> independent, moderate LER per repeat unit
                             (each periodic tile is genuinely distinguishable,
                             as in a reasonably well-controlled process).
                 "hard"   -> much smaller/longer-correlated LER and no
                             stochastic defects in the background, so
                             periodic repeat units are nearly indistinguishable
                             from one another -- this is the deliberately
                             adversarial "highly periodic array region" case
                             the problem statement asks you to analyze as an
                             honest failure mode.
    """
    rng = np.random.default_rng(seed)
    render_fn = render_dram_pattern if structure == "dram" else render_finfet_pattern

    # 1. Render the clean high-res reference site (this IS the
    #    "characterized location" -- its specific LER/defect realization
    #    is what makes it uniquely identifiable, not just its nominal grid).
    ref_clean = render_fn(
        size=size,
        pitch_px=ref_pitch_nm,
        line_width_px=ref_line_width_nm,
        contact_radius_px=ref_contact_r_nm,
        rng=rng,
    )

    # 2. Render an independent low-res periodic background covering the
    #    full search FOV (10x the physical area). Independent RNG stream
    #    -> visually similar repeat units, not identical to the reference.
    search_rng = np.random.default_rng(None if seed is None else seed + 999)
    search_pitch_px = ref_pitch_nm / zoom_ratio

    if difficulty == "hard":
        bg_ler_amplitude, bg_ler_correlation, bg_defect_prob = 0.08, 5.0, 0.0
    else:
        bg_ler_amplitude, bg_ler_correlation, bg_defect_prob = 0.5, 2.5, 0.03

    search_background = render_fn(
        size=size,
        pitch_px=search_pitch_px,
        line_width_px=ref_line_width_nm / zoom_ratio,
        contact_radius_px=ref_contact_r_nm / zoom_ratio,
        rng=search_rng,
        ler_amplitude_px=bg_ler_amplitude,
        ler_correlation_px=bg_ler_correlation,
        defect_prob=bg_defect_prob,
    )

    # 3. Downsample the *exact* reference pattern by the true zoom ratio
    #    (area-average, i.e. what an optical system integrating over a
    #    larger pixel footprint actually does) and embed it into the
    #    search background at a random grid-aligned location.
    patch = zoom(ref_clean, 1.0 / zoom_ratio, order=1)  # -> size/zoom_ratio px
    ph, pw = patch.shape

    margin = int(search_pitch_px * 2)
    max_xy = size - pw - margin
    top_left_x = int(rng.uniform(margin, max_xy))
    top_left_y = int(rng.uniform(margin, max_xy))

    search_clean = search_background.copy()
    search_clean[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw] = patch

    gt_x = top_left_x + pw / 2.0
    gt_y = top_left_y + ph / 2.0
    decoy_xy = None

    if difficulty == "hard":
        # Deliberately adversarial case: embed a SECOND copy of the exact
        # same clean patch at a different location -- this is the direct,
        # honest way to model "two dies that are genuinely, structurally
        # identical," rather than relying on noise alone to create
        # ambiguity. Each copy still gets independent per-pixel sensor
        # noise later (different location in the same noisy image), so
        # the two candidates end up close-but-not-identical in score --
        # exactly the periodic-array confusion the problem asks about.
        for _ in range(20):
            dx = int(rng.uniform(margin, max_xy))
            dy = int(rng.uniform(margin, max_xy))
            if abs(dx - top_left_x) > pw or abs(dy - top_left_y) > ph:
                decoy_xy = (dx, dy)
                break
        if decoy_xy is not None:
            dx, dy = decoy_xy
            search_clean[dy:dy + ph, dx:dx + pw] = patch

    # 4. Apply independent, imaging-pipeline noise. Search is noisier than
    #    reference per the problem FAQ.
    ref_noisy = apply_sem_noise(ref_clean, rng, photon_gain=120.0,
                                 read_noise_std=0.008, blur_sigma=0.6,
                                 scanline_std=0.008)
    search_noisy = apply_sem_noise(search_clean, search_rng, photon_gain=45.0,
                                    read_noise_std=0.02, blur_sigma=1.0,
                                    scanline_std=0.02)

    ground_truth = {
        "x": gt_x,
        "y": gt_y,
        "zoom_ratio": zoom_ratio,
        "patch_size_px": pw,
        "decoy_x": None if decoy_xy is None else decoy_xy[0] + pw / 2.0,
        "decoy_y": None if decoy_xy is None else decoy_xy[1] + ph / 2.0,
    }
    return ref_noisy, search_noisy, ground_truth


def to_uint8(img01):
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# BONUS: optical-microscope (RGB) generalization
# ----------------------------------------------------------------------
#
# Optical microscope images differ from SEM in ways with their own
# literature to cite:
#   - Diffraction-limited PSF blur (Abbe/Rayleigh criterion) -- generally
#     a LARGER blur than an SEM beam spot, and wavelength-dependent.
#   - Chromatic aberration -- R/G/B channels focus/magnify very slightly
#     differently through a real lens, so each channel gets a small,
#     independent sub-pixel shift.
#   - Color sensor noise -- photon shot noise per channel (Bayer-pattern
#     cameras have different pixel counts per color, but we approximate
#     with per-channel Poisson noise for simplicity), rather than SEM's
#     single-channel electron-count noise.
#   -> cite optical-microscopy PSF/aberration literature separately from
#      the SEM noise-model citations for your slide.

def _apply_chromatic_shift(img01, shift_xy):
    """Sub-pixel shift a channel via Fourier phase shift (band-limited,
    avoids interpolation blur artifacts from naive np.roll)."""
    dx, dy = shift_xy
    H, W = img01.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    shifted = np.fft.ifft2(np.fft.fft2(img01) * phase).real
    return shifted


def to_rgb_optical(img01, rng, blur_sigma=1.4, aberration_px=0.6, photon_gain=60.0):
    """Convert a clean grayscale pattern into a 3-channel optical-style
    image: per-channel diffraction blur + chromatic aberration shift +
    per-channel shot/read noise."""
    channels = []
    # Blue diffracts more (shorter wavelength => tighter Airy disk in
    # theory, but scattering/aberration effects often make blue *noisiest*
    # in practice at this abstraction level -- we keep it simple and only
    # vary the aberration shift per channel, which is the dominant,
    # well-documented effect).
    shifts = {
        "R": (aberration_px * 0.8, -aberration_px * 0.3),
        "G": (0.0, 0.0),  # green treated as the reference/focus channel
        "B": (-aberration_px * 0.9, aberration_px * 0.5),
    }
    for ch in ("R", "G", "B"):
        shifted = _apply_chromatic_shift(img01, shifts[ch])
        blurred = gaussian_filter(shifted, sigma=blur_sigma)
        lam = np.clip(blurred, 0, 1) * photon_gain
        shot = rng.poisson(lam).astype(np.float32) / photon_gain
        read = rng.normal(0, 0.01, size=img01.shape).astype(np.float32)
        channels.append(np.clip(shot + read, 0, 1))
    return np.stack(channels, axis=-1)  # H x W x 3, float32 in [0,1]


def generate_pair_rgb(size=1000, zoom_ratio=10, seed=None, structure="dram",
                       ref_pitch_nm=100.0, ref_line_width_nm=40.0, ref_contact_r_nm=15.0):
    """Same geometry/embedding logic as generate_pair, but rendered as
    RGB optical-microscope-style images instead of grayscale SEM (bonus
    generalization). Localization is done on the luminance channel by
    localize.localize_rgb, so no changes are needed to the core algorithm.
    """
    rng = np.random.default_rng(seed)
    render_fn = render_dram_pattern if structure == "dram" else render_finfet_pattern

    ref_clean = render_fn(size=size, pitch_px=ref_pitch_nm,
                           line_width_px=ref_line_width_nm,
                           contact_radius_px=ref_contact_r_nm, rng=rng)

    search_rng = np.random.default_rng(None if seed is None else seed + 999)
    search_pitch_px = ref_pitch_nm / zoom_ratio
    search_background = render_fn(size=size, pitch_px=search_pitch_px,
                                   line_width_px=ref_line_width_nm / zoom_ratio,
                                   contact_radius_px=ref_contact_r_nm / zoom_ratio,
                                   rng=search_rng, ler_amplitude_px=0.5,
                                   ler_correlation_px=2.5, defect_prob=0.03)

    patch = zoom(ref_clean, 1.0 / zoom_ratio, order=1)
    ph, pw = patch.shape
    margin = int(search_pitch_px * 2)
    top_left_x = int(rng.uniform(margin, size - pw - margin))
    top_left_y = int(rng.uniform(margin, size - pw - margin))
    search_clean = search_background.copy()
    search_clean[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw] = patch

    gt_x = top_left_x + pw / 2.0
    gt_y = top_left_y + ph / 2.0

    ref_rgb = to_rgb_optical(ref_clean, rng, photon_gain=100.0)
    search_rgb = to_rgb_optical(search_clean, search_rng, photon_gain=40.0)

    ground_truth = {"x": gt_x, "y": gt_y, "zoom_ratio": zoom_ratio, "patch_size_px": pw}
    return ref_rgb, search_rgb, ground_truth


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data", help="output directory")
    ap.add_argument("--n", type=int, default=5, help="number of pairs to generate")
    ap.add_argument("--seed", type=int, default=42, help="base random seed")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--structure", choices=["dram", "finfet"], default="dram")
    ap.add_argument("--difficulty", choices=["easy", "hard"], default="easy",
                     help="'hard' = low-LER, near-identical periodic repeats "
                          "(deliberately adversarial, for failure-case analysis)")
    ap.add_argument("--rgb", action="store_true",
                     help="generate RGB optical-microscope-style pairs instead "
                          "of grayscale SEM (bonus generalization)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_gt = {}

    for i in range(args.n):
        if args.rgb:
            ref, search, gt = generate_pair_rgb(size=args.size, seed=args.seed + i,
                                                  structure=args.structure)
        else:
            ref, search, gt = generate_pair(size=args.size, seed=args.seed + i,
                                              structure=args.structure,
                                              difficulty=args.difficulty)
        ref_name = f"pair_{i:03d}_reference.png"
        search_name = f"pair_{i:03d}_search.png"
        if args.rgb:
            Image.fromarray((np.clip(ref, 0, 1) * 255).astype(np.uint8)).save(os.path.join(args.out, ref_name))
            Image.fromarray((np.clip(search, 0, 1) * 255).astype(np.uint8)).save(os.path.join(args.out, search_name))
        else:
            Image.fromarray(to_uint8(ref)).save(os.path.join(args.out, ref_name))
            Image.fromarray(to_uint8(search)).save(os.path.join(args.out, search_name))
        all_gt[f"pair_{i:03d}"] = gt
        print(f"[{i}] ground truth center in search image: "
              f"x={gt['x']:.2f}, y={gt['y']:.2f}")

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(all_gt, f, indent=2)

    print(f"\nWrote {args.n} pairs + ground_truth.json to {args.out}")


if __name__ == "__main__":
    main()
