# Drift-Sense Phase 2 — Failure Analysis
**Team:** Yash Kothari, BITS Pilani | **Problem:** Applied Materials Drift-Sense

---

## 1. Hand-Rolled FFT-NCC Bug (Root Cause of Phase 1 Accuracy Failures)

**What failed:** The original `normalized_cross_correlation_fft` (integral image + FFT) had a genuine implementation bug. At the exact ground-truth (scale, theta), it returned wrong locations (18–358px error, score 0.66–0.82) while `cv2.matchTemplate` on the same inputs returned correct locations (0.3–0.7px error, score 0.85–0.87).

**Evidence:** Direct A/B comparison on real organizer sample pairs at known ground-truth pose. The hand-rolled implementation's score surface had systematic phase errors — it was not computing normalized cross-correlation correctly despite appearing to produce plausible-looking scores.

**Fix:** Replaced the core correlation primitive entirely with `cv2.matchTemplate(TM_CCOEFF_NORMED)`. All beam-search and GAR logic retained on top. Result: Set A=1.000, Set B=0.967, Set D=1.000, F1=1.000 vs organizer baseline A=1.000, B=0.467, D=1.000, F1=0.897.

---

## 2. Scale-Step Quantization Error

**What failed:** Stage 1 coarse grid used `scale_step=1.0` (grid: 8,9,10,11,12). For a true scale of e.g. 10.55, the nearest grid point is 11.0 — a 0.45 mismatch. At this off-scale template, the true embedded position's NCC score degraded enough that an unrelated structural region elsewhere in the search image scored higher by a clear margin. The beam search then refined the wrong branch confidently, producing 81px and 226px localization errors at high confidence (score≈0.85, GAR≈0.98).

**Evidence:** At exact true scale (10.55), the true position scored 0.8326 and won outright. At scale=11.0 (nearest grid), it scored only 0.5621 while an unrelated location scored 0.5875 — a real, not noise-level gap. Stage 2 theta refinement made it worse: score monotonically increased as theta moved *away* from true value (-2.3°→-1.0°), because the wrong location's match landscape was smoother and more favorable at integer scale.

**Fix:** Changed `scale_step=1.0` → `scale_step=0.5`. Stage 1 evaluations increased from 25 to 45 (9 scale × 5 theta anchors). Max scale quantization error reduced from 0.5 to 0.25. Timing impact: ~0.3s additional, negligible vs 17.5s budget. All seeds corrected to sub-1px error.

---

## 3. Absent-Pair Generator Design Flaw

**What failed:** `generate_dataset_v2.py`'s absent-pair branch embedded a same-pitch, same-architecture distractor patch into the search image, while leaving the reference as the true original template. This caused 100% false positive rate on locally-generated Set C: scores 0.62–0.85, all well above rejection threshold.

**Root cause:** Two renders of the same periodic pattern (same pitch, same architecture, different RNG seed) still correlate strongly under NCC — matchTemplate has 3 degrees of freedom (x, y, and scale/theta already fixed in stage3) to find the best phase alignment between two periodic grids. A sharp, confident false lock (score=0.70, GAR=0.33) resulted, not diffuse aliasing.

**Organizer's actual method** (confirmed from `phase2_pipeline.py` lines 380–413): search canvas left completely untouched; reference swapped for a decoy from an *independent* canvas with `mat_size×0.55`, `strip_width×2.1` — deliberately mismatched periodicity that structurally cannot occur anywhere in the true search image. Decoy also force-cropped onto a mat/strip junction for additional structural specificity.

**Fix:** Rewrote absent branch to mirror organizer exactly: `search_clean` left untouched, `ref_clean` replaced with decoy rendered at `pitch×0.55`, `line_width×2.1`, `contact_r×1.6`. Result: false positive rate 20/20→0/20, absent scores now 0.12–0.33, matching real organizer absent range (0.26–0.40).

---

## 4. Threshold Calibration — Borderline Case

**What failed:** Original thresholds `(score≥0.42 OR gar≤0.60)` produced 1 false positive on real organizer data: p018 (absent, Set C severity=0) scored 0.4291 — just 0.009 above threshold. The score gap between the closest present pair (p012, score=0.4292) and p018 was 0.0001 — numerically indistinguishable by score alone.

**Resolution:** GAR provides real separation: p012 GAR=0.6227 (confident, low ambiguity — true match), p018 GAR=0.7527 (more ambiguous — no true correspondence). GAR gap = 0.13, a meaningful margin. Changed thresholds to `(score≥0.43 OR gar≤0.65)`: both p012 and p018 now fail the score branch, but p012 passes the GAR branch (0.6227≤0.65) while p018 does not (0.7527>0.65). F1=1.000 with real margin on both axes.

---

## 5. Remaining Limitations

Local generator's search background shares pitch/architecture with the reference, causing occasional false locks (1/20) not seen on real data (GAR 0.45–0.86 real vs 0.98+ synthetic) — all thresholds calibrated on the real 20-pair set, not local data. Rejection margin is thin with only 4 real absent samples (p018=0.4291 vs hardest present p012=0.4292); GAR (0.13 gap) resolves it, but the real 200-pair set may include harder decoys per the organizer README, warranting re-check if more absent samples become available. Set D (RGB) handled via grayscale conversion; both pairs localized correctly (err<0.32px) but no dedicated multi-channel logic implemented.

---

## 6. Summary Table

| Issue | Impact | Fix | Status |
|-------|--------|-----|--------|
| FFT-NCC bug | Wrong location on all pairs | cv2.matchTemplate | Fixed |
| scale_step=1.0 | 81–226px errors, ~10% pairs | scale_step=0.5 | Fixed |
| Absent-pair generator | 100% false positive on local Set C | Mismatched-periodicity decoy | Fixed |
| Threshold calibration | 1 FP on real data (p018) | score≥0.43, gar≤0.65 | Fixed |
| Background self-similarity | Occasional local-data false locks | Generator realism issue; real data unaffected | Documented |
