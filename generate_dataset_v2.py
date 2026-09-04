"""
generate_dataset_v2.py
=======================
Phase 2 dataset generator for Drift-Sense. EXTENDS generate_dataset.py
(Phase 1) rather than replacing it -- reuses the same structure renderers
(render_finfet_pattern, render_dram_pattern) and noise model
(apply_sem_noise, to_rgb_optical) unchanged. This file only adds what
Phase 2 actually changed:

  - unknown zoom ratio, uniform in [8, 12]x (was exact 10x)
  - unknown rotation, uniform in [-5, +5] degrees (was small fixed noise)
  - ~20% "absent" pairs: a different-but-plausible periodic region is
    embedded instead of the true reference (found = 0)
  - degraded severity levels (0-3): charging artifacts, scan distortion,
    defocus, elevated shot noise, polygon (contact/line) scale jitter
  - RGB optical bonus set (reuses Phase 1's to_rgb_optical unchanged)

Default structure is now "finfet" (per team decision going into Phase 2);
"dram" remains fully supported via --structure.

Ground truth per pair: x, y, theta (degrees, CCW positive, about the match
centre), scale (recovered down-scaling factor), found (1/0).

USAGE
-----
    python3 generate_dataset_v2.py --out ./data_p2 --set A --n 70 --seed 100
    python3 generate_dataset_v2.py --out ./data_p2 --set B --n 70 --seed 200
    python3 generate_dataset_v2.py --out ./data_p2 --set C --n 40 --seed 300
    python3 generate_dataset_v2.py --out ./data_p2 --set D --n 20 --seed 400
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import zoom as ndi_zoom
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import map_coordinates

from generate_dataset import (
    render_finfet_pattern, render_dram_pattern,
    apply_sem_noise, to_rgb_optical, to_uint8,
)


# ----------------------------------------------------------------------
# Degraded-set effects (our own approximation of the disclosed categories:
# charging, scan distortion, defocus, elevated shot noise, polygon scale
# jitter -- exact organizer parameters are undisclosed by design, so this
# is for OUR OWN validation, not an attempt to reproduce their exact
# severity ladder)
# ----------------------------------------------------------------------

def apply_charging_artifact(img01, rng, severity):
    """Localized bright blooming near edges/insulating regions -- modeled
    as a smooth random-position Gaussian brightness bump, intensity and
    count scaled by severity (0-3)."""
    if severity == 0:
        return img01
    out = img01.copy()
    n_blobs = severity
    H, W = img01.shape
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        r = rng.uniform(40, 120)
        amp = 0.15 * severity
        bump = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        out = np.clip(out + bump, 0, 1)
    return out


def apply_scan_distortion(img01, rng, severity):
    """Smooth random 2D displacement field applied via map_coordinates --
    models raster-scan geometric instability (distinct from the per-row
    INTENSITY jitter already in apply_sem_noise, which is photometric,
    not geometric)."""
    if severity == 0:
        return img01
    H, W = img01.shape
    max_disp = 0.6 * severity  # pixels
    # low-frequency random field, upsampled to full resolution
    coarse = 8
    dy_coarse = rng.normal(0, max_disp, size=(coarse, coarse))
    dx_coarse = rng.normal(0, max_disp, size=(coarse, coarse))
    dy = ndi_zoom(dy_coarse, H / coarse, order=1)[:H, :W]
    dx = ndi_zoom(dx_coarse, W / coarse, order=1)[:H, :W]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = np.stack([yy + dy, xx + dx])
    return map_coordinates(img01, coords, order=1, mode="reflect")


def polygon_scale_jitter(size, pitch_px, line_width_px, contact_radius_px,
                          rng, severity, structure="finfet"):
    """Randomly perturb the nominal geometric parameters by up to +-20%
    at the given severity fraction, before rendering -- models CD
    (critical dimension) process variation at the polygon level, distinct
    from per-edge LER which is already handled inside the renderers."""
    frac = 0.20 * (severity / 3.0)
    jitter = lambda v: v * (1.0 + rng.uniform(-frac, frac))
    return jitter(pitch_px), jitter(line_width_px), jitter(contact_radius_px)


# ----------------------------------------------------------------------
# Core Phase 2 pair generator
# ----------------------------------------------------------------------

def generate_pair_v2(size=1000, seed=None, structure="finfet",
                      scale_range=(8.0, 12.0), rotation_range_deg=(-5.0, 5.0),
                      absent=False, degraded_severity=0,
                      ref_pitch_nm=100.0, ref_line_width_nm=40.0,
                      ref_contact_r_nm=15.0):
    """
    Returns (reference01, search01, ground_truth_dict).

    ground_truth_dict keys: x, y, theta_deg, scale, found, patch_size_px
    (x, y, theta_deg, scale are meaningful only if found == 1; by
    convention we still fill them with the "would-be" true values so a
    method's x,y,theta,scale predictions can be scored for reference, but
    the official contract only credits pose/localization when found==1
    AND was supposed to be 1).
    """
    rng = np.random.default_rng(seed)
    render_fn = render_finfet_pattern if structure == "finfet" else render_dram_pattern

    scale = float(rng.uniform(*scale_range))
    theta_deg = float(rng.uniform(*rotation_range_deg))

    # 1. Render the clean high-res reference site (1 nm/px).
    ref_clean = render_fn(size=size, pitch_px=ref_pitch_nm,
                           line_width_px=ref_line_width_nm,
                           contact_radius_px=ref_contact_r_nm, rng=rng)

    # 2. Render the periodic search background at (roughly) 10nm/px scale
    #    -- note: since the TRUE scale is unknown per-pair in [8,12]x, the
    #    background pitch is rendered at that pair's own true scale so the
    #    embedded patch and its surrounding look-alikes are geometrically
    #    consistent (same physical pitch, different LER noise realization).
    search_rng = np.random.default_rng(None if seed is None else seed + 999)
    search_pitch_px = ref_pitch_nm / scale
    p_pitch, p_width, p_contact = search_pitch_px, ref_line_width_nm / scale, ref_contact_r_nm / scale
    if degraded_severity > 0:
        p_pitch, p_width, p_contact = polygon_scale_jitter(
            size, p_pitch, p_width, p_contact, search_rng, degraded_severity, structure)

    search_background = render_fn(size=size, pitch_px=p_pitch, line_width_px=p_width,
                                   contact_radius_px=p_contact, rng=search_rng,
                                   ler_amplitude_px=0.5, ler_correlation_px=2.5,
                                   defect_prob=0.03)

    # 3. Build the patch to embed: downsample by the true scale, then
    #    rotate by the true angle (CCW positive, about the patch centre).
    #    reshape=True so the rotated bounding box isn't clipped; we track
    #    the resulting patch size for embedding + ground truth.
    patch_ds = ndi_zoom(ref_clean, 1.0 / scale, order=1)
    if abs(theta_deg) > 1e-6:
        patch = ndi_rotate(patch_ds, theta_deg, reshape=True, order=1, mode="constant", cval=0.0)
    else:
        patch = patch_ds
    ph, pw = patch.shape

    margin = int(search_pitch_px * 2) + 5
    max_xy = size - max(ph, pw) - margin
    if max_xy <= margin:
        max_xy = margin + 1  # degenerate small-image guard
    top_left_x = int(rng.uniform(margin, max_xy))
    top_left_y = int(rng.uniform(margin, max_xy))

    search_clean = search_background.copy()

    if not absent:
        # blend patch into background using its own mask (rotated patch
        # has zero-padding outside its rotated footprint; only overwrite
        # where the patch actually has rendered content OR simply
        # alpha-composite by taking the patch value where nonzero footprint)
        region = search_clean[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw]
        # footprint mask: valid (non-padding) rotated area
        ones = np.ones_like(patch_ds)
        footprint = ndi_rotate(ones, theta_deg, reshape=True, order=1,
                                mode="constant", cval=0.0) if abs(theta_deg) > 1e-6 else ones
        footprint = np.clip(footprint, 0, 1)
        blended = region * (1 - footprint) + patch * footprint
        search_clean[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw] = blended
        gt_x = top_left_x + pw / 2.0
        gt_y = top_left_y + ph / 2.0
    else:
        # ABSENT case (fixed to mirror the real organizer method, see
        # AMP generator src/phase2_pipeline.py): search_clean is left
        # COMPLETELY UNTOUCHED -- no embedding at all. Instead, the
        # REFERENCE itself is replaced with a decoy rendered from an
        # independent canvas at deliberately mismatched pitch/line-width
        # (organizer uses mat_size*0.55 / strip_width*2.1 for the same
        # reason). A same-pitch distractor embedded into the search image
        # (the old approach) lets the true reference still alias-match
        # against the untouched background at high confidence, since two
        # renders of the same periodic pitch correlate well regardless of
        # RNG seed once rotation/scale give matchTemplate freedom to hunt
        # for the best phase alignment -- confirmed empirically (score
        # 0.70, GAR 0.33, i.e. a sharp unambiguous false lock, not a
        # diffuse aliasing artifact). Mismatching pitch/line-width instead
        # ensures the decoy's periodicity structurally cannot occur
        # anywhere in the true search canvas.
        decoy_rng = np.random.default_rng(None if seed is None else seed + 555)
        decoy_pitch_nm = ref_pitch_nm * 0.55
        decoy_line_width_nm = ref_line_width_nm * 2.1
        decoy_contact_r_nm = ref_contact_r_nm * 1.6
        ref_clean = render_fn(size=size, pitch_px=decoy_pitch_nm,
                               line_width_px=decoy_line_width_nm,
                               contact_radius_px=decoy_contact_r_nm,
                               rng=decoy_rng)
        ph, pw = ref_clean.shape
        gt_x, gt_y = None, None  # no true location

    # 4. Degraded-set effects (geometric + photometric), applied to the
    #    CLEAN full search image before sensor noise.
    if degraded_severity > 0:
        search_clean = apply_charging_artifact(search_clean, search_rng, degraded_severity)
        search_clean = apply_scan_distortion(search_clean, search_rng, degraded_severity)

    # 5. Sensor noise (search noisier than reference; degraded sets get
    #    additionally elevated shot noise + defocus via lower photon_gain
    #    / larger blur_sigma).
    ref_noisy = apply_sem_noise(ref_clean, rng, photon_gain=120.0,
                                 read_noise_std=0.008, blur_sigma=0.6, scanline_std=0.008)
    degrade_gain_factor = 1.0 / (1.0 + 0.6 * degraded_severity)
    degrade_blur_bonus = 0.4 * degraded_severity
    search_noisy = apply_sem_noise(search_clean, search_rng,
                                    photon_gain=45.0 * degrade_gain_factor,
                                    read_noise_std=0.02 + 0.01 * degraded_severity,
                                    blur_sigma=1.0 + degrade_blur_bonus,
                                    scanline_std=0.02 + 0.01 * degraded_severity)

    ground_truth = {
        "x": gt_x, "y": gt_y,
        "theta_deg": theta_deg if not absent else None,
        "scale": scale if not absent else None,
        "found": 0 if absent else 1,
        "patch_size_px": pw,
        "structure": structure,
        "degraded_severity": degraded_severity,
    }
    return ref_noisy, search_noisy, ground_truth


# ----------------------------------------------------------------------
# RGB optical bonus (Set D) -- reuses Phase 1's to_rgb_optical unchanged
# ----------------------------------------------------------------------

def generate_pair_v2_rgb(size=1000, seed=None, structure="finfet",
                          scale_range=(8.0, 12.0), rotation_range_deg=(-5.0, 5.0),
                          ref_pitch_nm=100.0, ref_line_width_nm=40.0, ref_contact_r_nm=15.0):
    rng = np.random.default_rng(seed)
    render_fn = render_finfet_pattern if structure == "finfet" else render_dram_pattern

    scale = float(rng.uniform(*scale_range))
    theta_deg = float(rng.uniform(*rotation_range_deg))

    ref_clean = render_fn(size=size, pitch_px=ref_pitch_nm, line_width_px=ref_line_width_nm,
                           contact_radius_px=ref_contact_r_nm, rng=rng)
    search_rng = np.random.default_rng(None if seed is None else seed + 999)
    search_pitch_px = ref_pitch_nm / scale
    search_background = render_fn(size=size, pitch_px=search_pitch_px,
                                   line_width_px=ref_line_width_nm / scale,
                                   contact_radius_px=ref_contact_r_nm / scale,
                                   rng=search_rng, ler_amplitude_px=0.5,
                                   ler_correlation_px=2.5, defect_prob=0.03)

    patch_ds = ndi_zoom(ref_clean, 1.0 / scale, order=1)
    patch = ndi_rotate(patch_ds, theta_deg, reshape=True, order=1,
                        mode="constant", cval=0.0) if abs(theta_deg) > 1e-6 else patch_ds
    ph, pw = patch.shape
    margin = int(search_pitch_px * 2) + 5
    max_xy = size - max(ph, pw) - margin
    top_left_x = int(rng.uniform(margin, max_xy))
    top_left_y = int(rng.uniform(margin, max_xy))

    region = search_background[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw]
    ones = np.ones_like(patch_ds)
    footprint = ndi_rotate(ones, theta_deg, reshape=True, order=1,
                            mode="constant", cval=0.0) if abs(theta_deg) > 1e-6 else ones
    footprint = np.clip(footprint, 0, 1)
    search_clean = search_background.copy()
    search_clean[top_left_y:top_left_y + ph, top_left_x:top_left_x + pw] = (
        region * (1 - footprint) + patch * footprint)

    gt_x, gt_y = top_left_x + pw / 2.0, top_left_y + ph / 2.0

    ref_rgb = to_rgb_optical(ref_clean, rng, photon_gain=100.0)
    search_rgb = to_rgb_optical(search_clean, search_rng, photon_gain=40.0)

    ground_truth = {"x": gt_x, "y": gt_y, "theta_deg": theta_deg, "scale": scale,
                     "found": 1, "patch_size_px": pw, "structure": structure}
    return ref_rgb, search_rgb, ground_truth


# ----------------------------------------------------------------------
# Batch generation matching the organizers' Set A/B/C/D design
# ----------------------------------------------------------------------

SET_CONFIG = {
    "A": dict(n_default=70, absent_frac=0.0, degraded=False,
              desc="Nominal pose, reference present, full [8,12]x and +-5deg range"),
    "B": dict(n_default=70, absent_frac=0.0, degraded=True,
              desc="Degraded (charging/scan-distortion/defocus/noise/CD-jitter), 4 severity levels"),
    "C": dict(n_default=40, absent_frac=1.0, degraded=False,
              desc="Absent -- no true instance, plausible periodic distractor"),
    "D": dict(n_default=20, absent_frac=0.0, degraded=False, rgb=True,
              desc="Optical RGB bonus, reference present"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data_p2")
    ap.add_argument("--set", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--n", type=int, default=None, help="defaults to organizer's set size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structure", choices=["finfet", "dram"], default="finfet")
    ap.add_argument("--size", type=int, default=1000)
    args = ap.parse_args()

    cfg = SET_CONFIG[args.set]
    n = args.n or cfg["n_default"]
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for i in range(n):
        seed_i = args.seed + i
        rng_i = np.random.default_rng(seed_i + 424242)
        absent = rng_i.random() < cfg["absent_frac"]
        severity = int(rng_i.integers(0, 4)) if cfg["degraded"] else 0

        if cfg.get("rgb"):
            ref, search, gt = generate_pair_v2_rgb(size=args.size, seed=seed_i,
                                                     structure=args.structure)
            ref_img = Image.fromarray((np.clip(ref, 0, 1) * 255).astype(np.uint8))
            search_img = Image.fromarray((np.clip(search, 0, 1) * 255).astype(np.uint8))
        else:
            ref, search, gt = generate_pair_v2(size=args.size, seed=seed_i,
                                                 structure=args.structure,
                                                 absent=absent, degraded_severity=severity)
            ref_img = Image.fromarray(to_uint8(ref))
            search_img = Image.fromarray(to_uint8(search))

        pair_id = f"{args.set}_{i:03d}"
        ref_name = f"{pair_id}_reference.png"
        search_name = f"{pair_id}_search.png"
        ref_img.save(os.path.join(args.out, ref_name))
        search_img.save(os.path.join(args.out, search_name))

        rows.append({"pair_id": pair_id, "reference": ref_name, "search": search_name, **gt})

    with open(os.path.join(args.out, f"set_{args.set}_ground_truth.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # also write a pairs.csv in the exact shape register.py will receive,
    # for local end-to-end I/O contract testing
    import csv
    with open(os.path.join(args.out, f"set_{args.set}_pairs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "reference_path", "search_path"])
        for r in rows:
            writer.writerow([r["pair_id"],
                              os.path.join(args.out, r["reference"]),
                              os.path.join(args.out, r["search"])])

    n_absent = sum(1 for r in rows if r["found"] == 0)
    print(f"Set {args.set}: {cfg['desc']}")
    print(f"Generated {n} pairs ({n_absent} absent) -> {args.out}")
    print(f"  ground truth: set_{args.set}_ground_truth.json")
    print(f"  pairs.csv   : set_{args.set}_pairs.csv")


if __name__ == "__main__":
    main()
