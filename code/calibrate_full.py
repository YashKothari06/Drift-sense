"""
Full calibration rerun: captures score, gar, AND ground-truth correctness
together for all pairs across Set A/B/C, so we can properly threshold-sweep
found_score_thresh / found_gar_thresh / AND-vs-OR logic against real data,
rather than the earlier percentile-heuristic guess.

Writes calib_full_results.csv with one row per pair:
  set,pair_id,gt_found,score,gar,err_px (err_px only meaningful if gt_found=1)

Usage:
    python3 calibrate_full.py --dir calib_A --set A --out calib_full_results.csv
    python3 calibrate_full.py --dir calib_B --set B --out calib_full_results.csv --append
    python3 calibrate_full.py --dir calib_C --set C --out calib_full_results.csv --append
"""
import argparse, json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize_v2 import beam_search_localize

def load_image(path):
    from PIL import Image
    return np.array(Image.open(path).convert('L'), dtype=np.float64) / 255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--set', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--append', action='store_true')
    args = ap.parse_args()

    gt_path = os.path.join(args.dir, f'set_{args.set}_ground_truth.json')
    with open(gt_path) as f:
        gts = json.load(f)

    mode = 'a' if args.append and os.path.exists(args.out) else 'w'
    with open(args.out, mode, newline='') as f:
        w = csv.writer(f)
        if mode == 'w':
            w.writerow(['set', 'pair_id', 'gt_found', 'score', 'gar', 'err_px'])
        for i, gt in enumerate(gts):
            ref = load_image(os.path.join(args.dir, gt['reference']))
            search = load_image(os.path.join(args.dir, gt['search']))
            r = beam_search_localize(search, ref)
            if gt['found'] == 1:
                err = ((r['x'] - gt['x']) ** 2 + (r['y'] - gt['y']) ** 2) ** 0.5
            else:
                err = ''
            w.writerow([args.set, gt['pair_id'], gt['found'], r['score'], r['gar'], err])
            print(f"[{i+1}/{len(gts)}] {gt['pair_id']}: gt_found={gt['found']} score={r['score']:.3f} gar={r['gar']:.3f} err={err}")

if __name__ == '__main__':
    main()
