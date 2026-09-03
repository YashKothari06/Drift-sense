"""
Sweeps found_score_thresh x found_gar_thresh x {AND, OR} logic against
calib_full_results.csv to find the F1-maximizing combination, using the
data captured by calibrate_full.py.

Also reports the AUC-optimal single confidence metric (score, gar, or
score*(1-gar)) for the calibration axis.

Usage: python3 sweep_thresholds.py --csv calib_full_results.csv
"""
import argparse, csv
import numpy as np

def load(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(
                set=row['set'], pair_id=row['pair_id'],
                gt_found=int(row['gt_found']), score=float(row['score']),
                gar=float(row['gar']),
                err_px=float(row['err_px']) if row['err_px'] not in ('', None) else None))
    return rows

def f1_for_rule(rows, score_thresh, gar_thresh, logic):
    tp = fp = fn = tn = 0
    for r in rows:
        if logic == 'AND':
            pred = (r['score'] >= score_thresh) and (r['gar'] <= gar_thresh)
        elif logic == 'OR':
            pred = (r['score'] >= score_thresh) or (r['gar'] <= gar_thresh)
        elif logic == 'SCORE_ONLY':
            pred = r['score'] >= score_thresh
        elif logic == 'GAR_ONLY':
            pred = r['gar'] <= gar_thresh
        pred = int(pred)
        if r['gt_found'] == 1 and pred == 1: tp += 1
        elif r['gt_found'] == 0 and pred == 1: fp += 1
        elif r['gt_found'] == 1 and pred == 0: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall, tp, fp, fn, tn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    rows = load(args.csv)

    print(f"Loaded {len(rows)} pairs ({sum(r['gt_found'] for r in rows)} present, "
          f"{sum(1 - r['gt_found'] for r in rows)} absent)")

    score_grid = np.arange(0.1, 0.95, 0.025)
    gar_grid = np.arange(0.1, 0.99, 0.025)

    best = (0, None)
    results_by_logic = {}
    for logic in ['AND', 'OR', 'SCORE_ONLY', 'GAR_ONLY']:
        logic_best = (0, None)
        for st in score_grid:
            for gt_ in gar_grid:
                f1, prec, rec, tp, fp, fn, tn = f1_for_rule(rows, st, gt_, logic)
                if f1 > logic_best[0]:
                    logic_best = (f1, (st, gt_, prec, rec, tp, fp, fn, tn))
                if f1 > best[0]:
                    best = (f1, (logic, st, gt_, prec, rec, tp, fp, fn, tn))
        results_by_logic[logic] = logic_best

    print("\n--- Best per logic type ---")
    for logic, (f1, params) in results_by_logic.items():
        if params is None:
            continue
        st, gt_, prec, rec, tp, fp, fn, tn = params
        print(f"{logic}: F1={f1:.3f} (score_thresh={st:.3f}, gar_thresh={gt_:.3f}) "
              f"precision={prec:.3f} recall={rec:.3f} tp={tp} fp={fp} fn={fn} tn={tn}")

    print(f"\n=== OVERALL BEST ===")
    f1, (logic, st, gt_, prec, rec, tp, fp, fn, tn) = best
    print(f"logic={logic} found_score_thresh={st:.3f} found_gar_thresh={gt_:.3f}")
    print(f"F1={f1:.3f} precision={prec:.3f} recall={rec:.3f}")
    print(f"tp={tp} fp={fp} fn={fn} tn={tn}")

    def auc(scores, labels):
        scores = np.array(scores); labels = np.array(labels)
        if len(set(labels)) < 2:
            return None
        order = np.argsort(scores)
        ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores)+1)
        n_pos = labels.sum(); n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        return (ranks[labels==1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

    labels = []
    scores_raw, gars_raw, conf_raw = [], [], []
    for r in rows:
        if r['gt_found'] == 1:
            correct = 1 if (r['err_px'] is not None and r['err_px'] <= 5) else 0
        else:
            correct = 1 if r['gar'] > 0.9 else 0
        labels.append(correct)
        scores_raw.append(r['score'])
        gars_raw.append(1 - r['gar'])
        conf_raw.append(r['score'] * (1 - r['gar']))

    print(f"\n--- AUC of each raw signal vs (err_px<=5 for present, high-gar for absent) ---")
    print(f"score AUC:            {auc(scores_raw, labels)}")
    print(f"(1-gar) AUC:          {auc(gars_raw, labels)}")
    print(f"score*(1-gar) AUC:    {auc(conf_raw, labels)}")

if __name__ == '__main__':
    main()
