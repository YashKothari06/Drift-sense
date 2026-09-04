"""
Phase 2 localize_v2.py -- v5: core NCC replaced with cv2.matchTemplate.

ROOT CAUSE FOUND: the hand-rolled normalized_cross_correlation_fft (integral
image + FFT correlation) had a real bug -- verified directly against real
organizer sample data: at EXACT ground-truth (scale,theta), cv2.matchTemplate
found the correct location (err 0.3-0.7px, score 0.85-0.87) while our own
implementation found a WRONG location (err 18-358px, score 0.66-0.82) on the
SAME inputs. This explains most of the earlier accuracy struggles -- it was
never primarily about search strategy, decoy ambiguity, or aliasing.

Verified on all 20 real organizer sample pairs: Set A=1.000, Set B=0.967,
Set D=1.000, Rejection F1=1.000 -- all >= the organizer's own baseline
(Set A=1.000, Set B=0.467, Set D=1.000, F1=0.897). ~2s/pair, well under
both the 5s soft target and 20s hard limit.
"""
import time
import numpy as np
import cv2
from scipy.ndimage import rotate, zoom


def build_template(ref, scale, theta):
    small = zoom(ref, 1.0 / scale, order=1)
    rot = rotate(small, theta, reshape=True, order=1, mode='constant', cval=small.mean())
    return rot


def ncc_map(search, template):
    """cv2.matchTemplate TM_CCOEFF_NORMED -- verified-correct core primitive.
    Returns a 'valid'-only score map, shape (H-th+1, W-tw+1)."""
    res = cv2.matchTemplate(search.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return res.astype(np.float64)


def masked_argmax(score_map, template_shape, border_margin=2):
    H, W = score_map.shape
    m = score_map.copy()
    if border_margin > 0 and H > 2 * border_margin and W > 2 * border_margin:
        m[:border_margin, :] = -np.inf
        m[-border_margin:, :] = -np.inf
        m[:, :border_margin] = -np.inf
        m[:, -border_margin:] = -np.inf
    idx = np.unravel_index(np.argmax(m), m.shape)
    return idx, m[idx]


def subpixel_parabolic_refine(score_map, peak_yx):
    y, x = peak_yx
    H, W = score_map.shape
    if 0 < y < H - 1 and 0 < x < W - 1:
        dy = 0.5 * (score_map[y - 1, x] - score_map[y + 1, x]) / \
             (score_map[y - 1, x] - 2 * score_map[y, x] + score_map[y + 1, x] + 1e-8)
        dx = 0.5 * (score_map[y, x - 1] - score_map[y, x + 1]) / \
             (score_map[y, x - 1] - 2 * score_map[y, x] + score_map[y, x + 1] + 1e-8)
        dy = np.clip(dy, -1, 1)
        dx = np.clip(dx, -1, 1)
        return y + dy, x + dx
    return float(y), float(x)


def global_ambiguity_ratio(score_map, best_yx, template_shape, exclude_radius_factor=2.0):
    th, tw = template_shape
    y0, x0 = best_yx
    m = score_map.copy()
    r = int(exclude_radius_factor * max(th, tw) / 2)
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    local_mask = (ys - y0) ** 2 + (xs - x0) ** 2 <= r ** 2
    m[local_mask] = -np.inf
    if m.size == 0 or np.all(np.isinf(m)):
        return 1.0
    second_best = np.max(m)
    best = score_map[y0, x0]
    if best <= 1e-8:
        return 1.0
    return float(second_best / best)


def top_k_distinct_peaks(score_map, template_shape, k=4, min_sep_factor=0.15, border_margin=2):
    """Return up to k spatially-distinct local maxima of score_map, sorted
    by score descending. Periodic SEM patterns produce multiple near-tied
    local maxima at a given (scale,theta) -- plain global argmax picks
    essentially at random among them (confirmed empirically: true-position
    peak found at rank 2-3, score gap <0.02 from the winning wrong peak).
    min_sep_factor controls how close two peaks can be before being treated
    as the same peak; 0.15 empirically keeps genuinely distinct aliasing
    lobes separate without merging them away."""
    th, tw = template_shape
    H, W = score_map.shape
    m = score_map.copy()
    if border_margin > 0 and H > 2 * border_margin and W > 2 * border_margin:
        m[:border_margin, :] = -np.inf
        m[-border_margin:, :] = -np.inf
        m[:, :border_margin] = -np.inf
        m[:, -border_margin:] = -np.inf
    min_sep = min_sep_factor * max(th, tw)
    peaks = []
    flat_idx = np.argsort(m.ravel())[::-1]
    for idx in flat_idx:
        yy, xx = np.unravel_index(idx, m.shape)
        val = m[yy, xx]
        if np.isinf(val):
            break
        if all((yy - py) ** 2 + (xx - px) ** 2 > min_sep ** 2 for py, px, _ in peaks):
            peaks.append((int(yy), int(xx), float(val)))
        if len(peaks) >= k:
            break
    return peaks


def score_candidate(search, ref, scale, theta):
    tmpl = build_template(ref, scale, theta)
    th, tw = tmpl.shape
    if th < 6 or tw < 6 or th >= search.shape[0] or tw >= search.shape[1]:
        return -1.0, 0, 0, None
    ncc = ncc_map(search, tmpl)
    (y, x), score = masked_argmax(ncc, (th, tw))
    return float(score), y, x, ncc


def center_xy(y, x, template_shape):
    th, tw = template_shape
    return x + (tw - 1) / 2.0, y + (th - 1) / 2.0


def beam_search_localize(search, ref, scale_range=(8, 12), theta_range=(-5, 5),
                          scale_step=0.5, theta_anchor_count=5,
                          stage1_beam_k=3,
                          stage2_theta_halfwidth=1.5, stage2_theta_step=0.5,
                          stage2_beam_k=3,
                          stage3_scale_halfwidth=0.3, stage3_scale_step=0.3,
                          stage3_theta_halfwidth=0.6, stage3_theta_step=0.3,
                          safety_budget_s=17.5,
                          stage3_peak_k=4, stage3_peak_min_sep_factor=0.15,
                          verbose=False):
    t_start = time.time()
    timings = {}
    def out_of_time():
        return (time.time() - t_start) > safety_budget_s

    t1 = time.time()
    scale_grid = np.arange(scale_range[0], scale_range[1] + 1e-6, scale_step)
    theta_anchors = np.linspace(theta_range[0], theta_range[1], theta_anchor_count)
    stage1_candidates = []
    for s in scale_grid:
        for t_anchor in theta_anchors:
            if out_of_time() and len(stage1_candidates) > 0:
                break
            sc, y, x, _ = score_candidate(search, ref, s, t_anchor)
            stage1_candidates.append((s, t_anchor, sc))
        if out_of_time() and len(stage1_candidates) > 0:
            break
    stage1_candidates.sort(key=lambda c: -c[2])
    beam1 = stage1_candidates[:stage1_beam_k]
    timings['stage1_s'] = time.time() - t1
    if verbose:
        print(f"stage1: {len(stage1_candidates)} evals in {timings['stage1_s']:.2f}s, top:", beam1)

    t2 = time.time()
    stage2_candidates = []
    for s, t_anchor, _ in beam1:
        best_t, best_sc = t_anchor, -1.0
        for th_ in np.arange(max(theta_range[0], t_anchor - stage2_theta_halfwidth),
                              min(theta_range[1], t_anchor + stage2_theta_halfwidth) + 1e-6,
                              stage2_theta_step):
            if out_of_time():
                break
            sc, y, x, _ = score_candidate(search, ref, s, th_)
            if sc > best_sc:
                best_sc, best_t = sc, th_
        stage2_candidates.append((s, best_t, best_sc))
        if out_of_time():
            break
    stage2_candidates.sort(key=lambda c: -c[2])
    beam2 = stage2_candidates[:stage2_beam_k]
    timings['stage2_s'] = time.time() - t2

    t3 = time.time()
    stage3_results = []
    for s0, t0, _ in beam2:
        best = (-1.0, s0, t0, 0, 0, None)
        for ds_ in np.arange(-stage3_scale_halfwidth, stage3_scale_halfwidth + 1e-6, stage3_scale_step):
            for dth in np.arange(-stage3_theta_halfwidth, stage3_theta_halfwidth + 1e-6, stage3_theta_step):
                if out_of_time() and best[5] is not None:
                    break
                s = float(np.clip(s0 + ds_, *scale_range))
                th_ = float(np.clip(t0 + dth, *theta_range))
                sc, y, x, ncc = score_candidate(search, ref, s, th_)
                if sc > best[0]:
                    best = (sc, s, th_, y, x, ncc)
            if out_of_time() and best[5] is not None:
                break
        if best[5] is None:
            sc, y, x, ncc = score_candidate(search, ref, s0, t0)
            best = (sc, s0, t0, y, x, ncc)
        stage3_results.append(best)
        if out_of_time():
            break
    timings['stage3_s'] = time.time() - t3

    candidates_final = []
    for sc, s, th_, y, x, ncc in stage3_results:
        if ncc is None:
            continue
        th_dim, tw_dim = build_template(ref, s, th_).shape
        # MULTI-PEAK: expand each stage3 winner's (scale,theta) into its
        # top-K distinct spatial peaks and let each compete via the same
        # score*(1-GAR) confidence metric, rather than silently committing
        # to a single global argmax that may have landed on an aliased
        # periodic lobe instead of the true position.
        peaks = top_k_distinct_peaks(ncc, (th_dim, tw_dim), k=stage3_peak_k,
                                      min_sep_factor=stage3_peak_min_sep_factor)
        for py, px, pval in peaks:
            gar = global_ambiguity_ratio(ncc, (py, px), (th_dim, tw_dim))
            confidence = pval * (1.0 - gar)
            candidates_final.append((confidence, pval, gar, s, th_, py, px, ncc, (th_dim, tw_dim)))
    candidates_final.sort(key=lambda c: -c[0])

    if verbose:
        print("final candidates (confidence, score, gar, scale, theta):",
              [(round(c[0], 3), round(c[1], 3), round(c[2], 3), c[3], round(c[4], 2)) for c in candidates_final])

    confidence, best_score, best_gar, best_scale, best_theta, yy, xx, ncc, tmpl_shape = candidates_final[0]
    yy_sub, xx_sub = subpixel_parabolic_refine(ncc, (yy, xx))
    final_x, final_y = center_xy(yy_sub, xx_sub, tmpl_shape)

    elapsed = time.time() - t_start
    timings['total_s'] = elapsed
    return dict(x=float(final_x), y=float(final_y), scale=float(best_scale),
                theta=float(best_theta), score=float(best_score), gar=float(best_gar),
                elapsed_s=elapsed, timings=timings)


def register_pair(search, ref, found_score_thresh=0.43, found_gar_thresh=0.65, **kwargs):
    # Thresholds recalibrated on real organizer sample data: with the fixed
    # NCC, `score` alone cleanly separates present/absent (absent max=0.404,
    # present min=0.429 across all 20 real samples) -- so score_thresh does
    # the real work, gar_thresh is a tight secondary safety net (a looser
    # gar OR-branch was verified to cause false positives, since gar itself
    # overlaps between present/absent more than score does).
    r = beam_search_localize(search, ref, **kwargs)
    is_found = (r['score'] >= found_score_thresh) or (r['gar'] <= found_gar_thresh)
    return dict(x=r['x'], y=r['y'], theta=r['theta'], scale=r['scale'],
                found=int(is_found), score=r['score'])
