#!/usr/bin/env python3
"""
Local self-assessment scorer mirroring the official Phase 2 rubric.

Usage:
    python3 scorer.py --gt-a set_A_ground_truth.json --gt-b set_B_ground_truth.json \
                       --gt-c set_C_ground_truth.json --pred-a predictions_A.csv \
                       --pred-b predictions_B.csv --pred-c predictions_C.csv \
                       --median-time-s 12.5

ASSUMPTIONS (rubric doc didn't give exact point splits for these -- flagged
so you know what's real spec vs my best-guess interpretation):
  1. Pose recovery tiers: I mirrored the localization tier SHAPE (full credit
     at tightest tier, decreasing, zero beyond loosest) since the addendum
     only gave the threshold values (1%/2%/5% for scale, 0.25/0.5/1.0deg for
     rotation) not the point split between tiers. Used: <=tier1: 1.0,
     <=tier2: 0.7, <=tier3: 0.4, else 0 -- SAME shape as localization's own
     4-tier scheme collapsed to 3 tiers since only 3 thresholds were given.
     If the real rubric differs, only this function needs editing.
  2. Confidence calibration (AUC): defined "correct" as found==1 AND
     loc_err<=5px for present pairs, or found==0 for absent pairs -- i.e.
     does `score` distinguish genuinely good outcomes from bad ones. This
     is my interpretation of "internal monotonicity", not verified against
     the official definition.
  3. Efficiency (5pts, quartile ranking): CANNOT be computed locally --
     needs other teams' timings. This script reports your median wall-clock
     only, as a diagnostic, and assigns 0 pts (worst case) as a
     conservative placeholder in the total.
  4. Set D (RGB bonus, +10pts) is not scored here -- register.py doesn't
     yet handle RGB input; out of scope for this pass.
"""
import argparse
import json
import csv
import numpy as np


def load_gt(path):
    with open(path) as f:
        return {row['pair_id']: row for row in json.load(f)}


def load_pred(path):
    with open(path) as f:
        return {row['pair_id']: row for row in csv.DictReader(f)}


def loc_credit(err_px):
    if err_px <= 1: return 1.0
    if err_px <= 2: return 0.8
    if err_px <= 3: return 0.6
    if err_px <= 5: return 0.4
    return 0.0


def pose_credit_scale(pct_err):
    if pct_err <= 1: return 1.0
    if pct_err <= 2: return 0.7
    if pct_err <= 5: return 0.4
    return 0.0


def pose_credit_theta(deg_err):
    if deg_err <= 0.25: return 1.0
    if deg_err <= 0.5: return 0.7
    if deg_err <= 1.0: return 0.4
    return 0.0


def evaluate_set(gt, pred):
    """Returns list of per-pair dicts with all raw metrics for one set."""
    rows = []
    for pid, g in gt.items():
        p = pred.get(pid)
        if p is None:
            rows.append(dict(pair_id=pid, missing=True, gt_found=g['found']))
            continue
        pred_found = int(p['found'])
        row = dict(pair_id=pid, missing=False, gt_found=g['found'], pred_found=pred_found,
                   pred_score=float(p['score']))
        if g['found'] == 1:
            err = ((float(p['x']) - g['x']) ** 2 + (float(p['y']) - g['y']) ** 2) ** 0.5
            # Match real score_baseline.py: loc_credit is ZERO whenever the
            # method predicted found=0, regardless of how close the
            # would-be position was -- a missed detection earns no partial
            # localization credit. Confirmed against organizer's own
            # `cr = credit(err) if pred_present else 0.0`.
            lc = loc_credit(err) if pred_found == 1 else 0.0
            scale_pct_err = abs(float(p['scale']) - g['scale']) / g['scale'] * 100
            theta_err = abs(float(p['theta']) - g['theta_deg'])
            pc_scale = pose_credit_scale(scale_pct_err) if lc > 0 else 0.0
            pc_theta = pose_credit_theta(theta_err) if lc > 0 else 0.0
            row.update(err_px=err, loc_credit=lc, scale_pct_err=scale_pct_err,
                       theta_err=theta_err, pose_credit_scale=pc_scale, pose_credit_theta=pc_theta)
        rows.append(row)
    return rows


def compute_f1(all_rows):
    tp = fp = fn = tn = 0
    for r in all_rows:
        if r.get('missing'):
            pred_found = 0
        else:
            pred_found = r['pred_found']
        gt_found = r['gt_found']
        if gt_found == 1 and pred_found == 1: tp += 1
        elif gt_found == 0 and pred_found == 1: fp += 1
        elif gt_found == 1 and pred_found == 0: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def compute_auc(all_rows):
    """AUC of pred_score distinguishing 'correct' outcomes from 'incorrect'.
    See module docstring assumption #2 for the definition of 'correct' used."""
    scores, labels = [], []
    for r in all_rows:
        if r.get('missing'):
            continue
        s = r['pred_score']
        if r['gt_found'] == 1:
            correct = 1 if (r.get('pred_found') == 1 and r.get('err_px', 1e9) <= 5) else 0
        else:
            correct = 1 if r.get('pred_found') == 0 else 0
        scores.append(s)
        labels.append(correct)
    scores = np.array(scores)
    labels = np.array(labels)
    if len(set(labels)) < 2:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-a', required=True)
    ap.add_argument('--gt-b', required=True)
    ap.add_argument('--gt-c', required=True)
    ap.add_argument('--pred-a', required=True)
    ap.add_argument('--pred-b', required=True)
    ap.add_argument('--pred-c', required=True)
    ap.add_argument('--median-time-s', type=float, default=None,
                     help='your measured median wall-clock/pair, for the efficiency diagnostic')
    args = ap.parse_args()

    gt_a, gt_b, gt_c = load_gt(args.gt_a), load_gt(args.gt_b), load_gt(args.gt_c)
    pred_a, pred_b, pred_c = load_pred(args.pred_a), load_pred(args.pred_b), load_pred(args.pred_c)

    rows_a = evaluate_set(gt_a, pred_a)
    rows_b = evaluate_set(gt_b, pred_b)
    rows_c = evaluate_set(gt_c, pred_c)
    all_rows = rows_a + rows_b + rows_c

    def mean_loc_credit(rows):
        present = [r for r in rows if not r.get('missing') and r['gt_found'] == 1]
        if not present:
            return 0.0, 0
        return np.mean([r['loc_credit'] for r in present]), len(present)

    loc_a, n_a = mean_loc_credit(rows_a)
    loc_b, n_b = mean_loc_credit(rows_b)
    loc_score = 40 * (0.45 * loc_a + 0.55 * loc_b)

    scored_present = [r for r in all_rows if not r.get('missing') and r.get('gt_found') == 1 and r.get('loc_credit', 0) > 0]
    if scored_present:
        pose_scale_score = 10 * np.mean([r['pose_credit_scale'] for r in scored_present])
        pose_theta_score = 10 * np.mean([r['pose_credit_theta'] for r in scored_present])
    else:
        pose_scale_score = pose_theta_score = 0.0
    pose_score = pose_scale_score + pose_theta_score

    f1_stats = compute_f1(all_rows)
    rejection_score = 15 * f1_stats['f1']

    auc = compute_auc(all_rows)
    calibration_score = 10 * auc if auc is not None else None

    efficiency_score = 0.0

    total_measurable = loc_score + pose_score + rejection_score + (calibration_score or 0)

    print("=" * 70)
    print("LOCAL SELF-ASSESSMENT (see docstring for assumptions/caveats)")
    print("=" * 70)
    print(f"\n--- Localization (40 pts, 0.45*A + 0.55*B) ---")
    print(f"  Set A: n={n_a} present pairs, mean_loc_credit={loc_a:.3f}")
    print(f"  Set B: n={n_b} present pairs, mean_loc_credit={loc_b:.3f}")
    print(f"  Score: {loc_score:.2f} / 40")

    print(f"\n--- Pose recovery (20 pts, scale+rotation, ASSUMED tier split) ---")
    print(f"  n={len(scored_present)} pairs with loc_credit>0")
    print(f"  Scale sub-score: {pose_scale_score:.2f} / 10")
    print(f"  Rotation sub-score: {pose_theta_score:.2f} / 10")
    print(f"  Score: {pose_score:.2f} / 20")

    print(f"\n--- Rejection F1 (15 pts, across {len(all_rows)} pairs) ---")
    print(f"  tp={f1_stats['tp']} fp={f1_stats['fp']} fn={f1_stats['fn']} tn={f1_stats['tn']}")
    print(f"  precision={f1_stats['precision']:.3f} recall={f1_stats['recall']:.3f} f1={f1_stats['f1']:.3f}")
    print(f"  Score: {rejection_score:.2f} / 15")

    print(f"\n--- Confidence calibration (10 pts, AUC, ASSUMED 'correct' definition) ---")
    if auc is not None:
        print(f"  AUC={auc:.3f}")
        print(f"  Score: {calibration_score:.2f} / 10")
    else:
        print("  Could not compute (only one outcome class present in this sample)")

    print(f"\n--- Efficiency (5 pts, quartile rank vs other teams -- NOT computable locally) ---")
    if args.median_time_s:
        print(f"  Your median wall-clock/pair: {args.median_time_s:.2f}s (target <=5s, hard limit 20s)")
    print(f"  Score: 0.00 / 5 (conservative placeholder -- depends on other teams' submissions)")

    print(f"\n--- Carried forward (10 pts: generator + citations + failure analysis) ---")
    print(f"  Not assessed by this script -- human-judged component")

    print("\n" + "=" * 70)
    print(f"MEASURABLE TOTAL (excl. efficiency + carried-forward): {total_measurable:.2f} / 85")
    print(f"With 0 assumed for efficiency: {total_measurable:.2f} / 90 possible from automated axes")
    print("=" * 70)


if __name__ == '__main__':
    main()
