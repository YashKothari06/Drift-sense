"""
Calibrate found_score_thresh / found_gar_thresh against YOUR real
generate_dataset_v2.py Set A (present) and Set C (absent) output.

Usage:
    python3 generate_dataset_v2.py --out ./calib_A --set A --n 20 --seed 1 --structure finfet
    python3 generate_dataset_v2.py --out ./calib_C --set C --n 20 --seed 2 --structure finfet
    python3 calibrate_threshold.py --dir ./calib_A --set A
    python3 calibrate_threshold.py --dir ./calib_C --set C
    (run both, then look at the combined suggestion printed at the end of the
    second call's --other-dir, OR just pass both dirs at once -- see --dir2 below)
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize_v2 import beam_search_localize

def load_image(path):
    from PIL import Image
    return np.array(Image.open(path).convert('L'), dtype=np.float64) / 255.0

def gather(set_dir, set_name):
    gt_path = os.path.join(set_dir, f'set_{set_name}_ground_truth.json')
    with open(gt_path) as f:
        gts = json.load(f)
    scores, gars, founds = [], [], []
    for gt in gts:
        ref = load_image(os.path.join(set_dir, gt['reference']))
        search = load_image(os.path.join(set_dir, gt['search']))
        r = beam_search_localize(search, ref)
        scores.append(r['score']); gars.append(r['gar']); founds.append(gt['found'])
        print(gt['pair_id'], 'gt_found=', gt['found'], 'score=', round(r['score'], 3),
              'gar=', round(r['gar'], 3),
              'loc_err=', round(((r['x']-gt['x'])**2 + (r['y']-gt['y'])**2)**0.5, 2) if gt['found'] else 'n/a')
    return np.array(scores), np.array(gars), np.array(founds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='set A dir (present pairs)')
    ap.add_argument('--dir2', required=True, help='set C dir (absent pairs)')
    args = ap.parse_args()

    print("=== Set A (present) ===")
    scores_a, gars_a, founds_a = gather(args.dir, 'A')
    print("\n=== Set C (absent) ===")
    scores_c, gars_c, founds_c = gather(args.dir2, 'C')

    scores_present = scores_a[founds_a == 1]
    scores_absent = scores_c[founds_c == 0]
    gars_present = gars_a[founds_a == 1]
    gars_absent = gars_c[founds_c == 0]

    print("\n--- score distributions ---")
    print("present p10/p50/p90:", np.percentile(scores_present, [10, 50, 90]) if len(scores_present) else 'none')
    print("absent  p10/p50/p90:", np.percentile(scores_absent, [10, 50, 90]) if len(scores_absent) else 'none')
    print("\n--- GAR distributions ---")
    print("present p10/p50/p90:", np.percentile(gars_present, [10, 50, 90]) if len(gars_present) else 'none')
    print("absent  p10/p50/p90:", np.percentile(gars_absent, [10, 50, 90]) if len(gars_absent) else 'none')

    if len(scores_present) and len(scores_absent):
        score_suggestion = (np.percentile(scores_absent, 90) + np.percentile(scores_present, 10)) / 2
        print(f"\nSuggested found_score_thresh: {score_suggestion:.3f}")
        print("  -> edit found_score_thresh default in localize_v2.py's register_pair()")
    if len(gars_present) and len(gars_absent):
        gar_suggestion = (np.percentile(gars_absent, 10) + np.percentile(gars_present, 90)) / 2
        print(f"Suggested found_gar_thresh: {gar_suggestion:.3f}")
        print("  -> edit found_gar_thresh default in localize_v2.py's register_pair()")

if __name__ == '__main__':
    main()
