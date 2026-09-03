# Drift-Sense — Semicon India Hackathon Submission

**AI-powered navigation-error recovery for wafer inspection tools.**
Semicon India Hackathon — Applied Materials Problem Statement 2

**Author:** Yash Kothari, BITS Pilani (Electronics)

---

## Overview

Drift-Sense addresses the navigation-error recovery problem in semiconductor wafer inspection: given a high-resolution reference image (1000×1000, 1 nm/px, 100× zoom) of a previously characterized site and a lower-resolution search image (1000×1000, ~10 nm/px, 10× zoom), locate the reference pattern's center inside the search image to sub-pixel precision — despite periodic-array ambiguity making this a genuine needle-in-a-haystack problem.

**Phase 1** delivered a classical FFT-NCC matched-filtering pipeline with a novel prior-window search restriction achieving 100% success at sub-pixel accuracy.

**Phase 2** extends this to handle unknown zoom (8–12×), unknown rotation (±5°), absent-pair rejection, and degraded imaging conditions — using a multi-stage beam search over (scale, θ, position) with `cv2.matchTemplate`, beating the organizer's own baseline on all scoring axes.

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, pillow, opencv-python-headless, torch (torch needed only for Phase 1 DL ablation — not used at Phase 2 inference).

---

## Phase 2 — Quick Start

```bash
cd code

# Run on a pairs.csv file (organizer format):
python3 register.py --input pairs.csv --output predictions.csv

# Output columns: pair_id, x, y, theta, scale, found, score
```

**`register.py`** is the submission entry point. It:
1. Reads a CSV of (pair_id, reference_path, search_path) rows
2. Runs `localize_v2.beam_search_localize()` on each pair
3. Applies rejection logic (`found = score≥0.43 OR gar≤0.65`)
4. Enforces a 20s hard timeout per pair via subprocess isolation
5. Writes predictions CSV

### Self-Assessment

```bash
python3 scorer.py \
  --gt-a gt_real_A.json --gt-b gt_real_B.json --gt-c gt_real_C.json \
  --pred-a pred_A.csv --pred-b pred_B.csv --pred-c pred_C.csv \
  --median-time-s 2.1
```

### Generate Synthetic Phase 2 Data

```bash
python3 generate_dataset_v2.py --out ./data_p2 --set A --n 70 --seed 100
python3 generate_dataset_v2.py --out ./data_p2 --set B --n 70 --seed 200
python3 generate_dataset_v2.py --out ./data_p2 --set C --n 40 --seed 300
python3 generate_dataset_v2.py --out ./data_p2 --set D --n 20 --seed 400
```

### Threshold Calibration

```bash
python3 calibrate_full.py --dir calib_A --set A --out calib_full_results.csv
python3 calibrate_full.py --dir calib_B --set B --out calib_full_results.csv --append
python3 calibrate_full.py --dir calib_C --set C --out calib_full_results.csv --append
python3 sweep_thresholds.py --csv calib_full_results.csv
```

---

## Phase 1 — Original Pipeline

```bash
cd code
python3 generate_dataset.py --out ./data --n 4 --seed 42
python3 localize.py --reference data/pair_000_reference.png --search data/pair_000_search.png --zoom 10
python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty easy
python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty hard
python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty hard --use_prior --max_drift_px 60
python3 dl_localize.py --train --epochs 800 --size 300 --max_seconds 250 --out siamese_300.pt --seed 0
python3 dl_localize.py --eval --weights siamese_300.pt --size 300 --n 30 --tolerance 3.0 --seed 0 --difficulty hard --refine --use_prior --max_drift_px 60
```

---

## Folder Structure

```
code/
  register.py              Phase 2 CLI entry point (organizer runs this)
  localize_v2.py            Phase 2 beam-search NCC localizer (cv2.matchTemplate)
  generate_dataset_v2.py    Phase 2 synthetic generator (extends Phase 1)
  scorer.py                 Local self-assessment scorer (mirrors official rubric)
  calibrate_threshold.py    Threshold calibration tool
  calibrate_full.py         Full calibration rerun tool
  sweep_thresholds.py       Threshold sweep optimizer
  generate_dataset.py       Phase 1 DRAM/FinFET/RGB dataset generator
  localize.py               Phase 1 classical FFT-NCC localizer
  evaluate.py               Phase 1 batch evaluation harness
  dl_localize.py            Phase 1 DL ablation (Siamese CNN)

docs/
  proposal.tex / .pdf       Phase 1 proposal (Round 1 submission)
  citations.md              Full annotated bibliography (15+ sources)

sample_outputs/
  eval_results*.json        Phase 1 evaluation results
  failure_case_seed24.*     Documented failure case (explainability)
  finfet_*, optical_*       Example SEM / optical images
  siamese_300.pt            Trained DL weights (Phase 1 ablation)
  pred_real_v2.csv          Phase 2 predictions on organizer sample data

requirements.txt            Python dependencies
```

---

## Headline Results

### Phase 2 — Verified on Real Organizer Sample Data (20 pairs)

| Set | Description | Accuracy | Organizer Baseline |
|---|---|---|---|
| Set A (8 pairs) | Nominal pose | **8/8 (1.000)** | 1.000 |
| Set B (6 pairs) | Degraded (charging, defocus, noise) | **6/6 (1.000)** | 0.467 |
| Set C (4 pairs) | Absent — rejection | **4/4, F1=1.000** | F1=0.897 |
| Set D (2 pairs) | RGB optical bonus | **2/2 (1.000)** | 1.000 |

- Median localization error: **0.26 px** (sub-pixel)
- Median wall-clock: **2.13 s/pair** (budget: ≤5s target, 20s hard limit)
- All errors within tier 1 (≤1px) except one tier 2 (1.18px)

### Phase 1 — Classical FFT-NCC (30 randomized cases)

| Configuration | Success Rate | Mean Error | Compute Time |
|---|---|---|---|
| Non-adversarial (easy) | 100% (30/30) | 0.03 px | ~309 ms |
| Adversarial decoy, no prior | 46.7% (14/30) | 195 px | ~224 ms |
| **Adversarial decoy + prior window** | **100% (30/30)** | **0.02 px** | **~215 ms** |

### Phase 1 — DL Ablation (Siamese CNN, 30 adversarial cases)

| Configuration | Success Rate | Mean Error | Compute Time |
|---|---|---|---|
| Raw DL argmax | 6.7% | 73.6 px | 4.4 ms |
| DL + NCC refinement | 50.0% | 66.1 px | 7.1 ms |
| DL + refinement + prior | 86–90% | ~5–8 px | 10–49 ms |

---

## Technical Approach

### Phase 2 Algorithm — Multi-Stage Beam Search

1. **Stage 1 (Coarse):** Sweep 9 scale values × 5 θ anchors = 45 evaluations. Each: `cv2.matchTemplate(TM_CCOEFF_NORMED)`. Keep top-3 by score.
2. **Stage 2 (θ Refinement):** Per beam candidate, sweep θ ±1.5° in 0.5° steps. Keep top-3.
3. **Stage 3 (Fine Joint):** Per beam candidate, sweep scale ±0.3 and θ ±0.6° simultaneously. Extract top-4 distinct spatial peaks per NCC surface.
4. **Selection:** `confidence = score × (1 − GAR)`. Subpixel parabolic refinement. Rejection: `found = (score ≥ 0.43) OR (gar ≤ 0.65)`.

### Phase 1 Algorithm — FFT-NCC + Prior Window

1. Downsample reference by known 10× zoom ratio
2. FFT-based normalized cross-correlation (Lewis, 1995)
3. Prior-window search restriction (bounded drift around commanded position)
4. Subpixel refinement via parabolic interpolation
5. GAR (Global Ambiguity Ratio) for periodic-decoy detection

### Key Innovation: Prior-Window Search Restriction

The tool knows where it commanded the stage — drift is a small, bounded error, not an arbitrary location. Restricting search to a window around the expected coordinate is simultaneously faster AND more robust to periodic decoys, with zero loss of precision. Grounded in US Patent 7,545,497.

---

## Literature Justification

Every noise/augmentation/algorithmic choice is grounded against public sources. Full annotated bibliography: [`docs/citations.md`](docs/citations.md).

| Design Choice | Source Category |
|---|---|
| Poisson-Gaussian SEM noise | Avci et al.; ScienceDirect M-Denoiser (2023); Oxford Academic (2025) |
| Line-edge roughness (LER/LWR) | Bunday/Bishop/Villarrubia (NIST/SEMATECH); ITRS targets |
| FinFET pitch/width | IEEE "Scaling of SOI FinFETs" |
| Optical chromatic aberration | Nikon MicroscopyU; Wadsworth Center |
| Fast NCC via FFT | Lewis, "Fast Normalized Cross-Correlation" (1995) |
| Peak-to-Sidelobe Ratio | Bolme et al. MOSSE tracker, CVPR 2010 |
| Prior-window restriction | US Patent 7,545,497; arXiv 2012.12784; Patent 12,327,739 |

---

## Key Bugs Found & Fixed (Phase 2)

1. **FFT-NCC Implementation Bug:** Hand-rolled FFT-NCC returned wrong locations (18–358px error) even at exact ground-truth pose. Replaced with `cv2.matchTemplate(TM_CCOEFF_NORMED)` — verified correct on all 20 organizer pairs.
2. **Scale-Step Quantization:** `scale_step=1.0` too coarse for [8–12×] range → `scale_step=0.5` resolves.
3. **Absent-Pair Generator:** Same-pitch distractors alias against true reference → switched to mismatched-periodicity decoy reference (matching organizer design).
4. **Threshold Calibration:** GAR provides critical secondary separation for borderline-score pairs (0.13 real margin vs 0.0001 score gap).

Full methodology, literature justification, and honest failure-case analysis are in [`docs/proposal.pdf`](docs/proposal.pdf).
