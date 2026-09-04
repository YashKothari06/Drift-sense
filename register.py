#!/usr/bin/env python3
"""
Phase 2 CLI: python register.py --input pairs.csv --output predictions.csv

Enforces the 20s hard timeout per pair via a subprocess worker (so a hang,
not just a slow-but-alive loop, still gets caught) -- a timed-out or crashed
pair is written as found=0 rather than aborting the whole run.

ASSUMPTION TO VERIFY: this auto-detects likely column names for the
reference/search image paths in your pairs.csv. Run:
    head -3 pairs.csv
and check the printed "detected columns" line on first run matches what you
expect -- if not, pass --ref-col / --search-col / --id-col explicitly.
"""
import argparse
import csv
import multiprocessing as mp
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localize_v2 import register_pair

TIMEOUT_S = 20.0


def load_image(path):
    from PIL import Image
    img = np.array(Image.open(path).convert('L'), dtype=np.float64) / 255.0
    return img


def _worker(ref_path, search_path, q):
    try:
        ref = load_image(ref_path)
        search = load_image(search_path)
        r = register_pair(search, ref)
        q.put(('ok', r))
    except Exception as e:
        q.put(('err', str(e)))


def run_one_pair(ref_path, search_path):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(ref_path, search_path, q))
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(x=0.0, y=0.0, theta=0.0, scale=10.0, found=0, score=0.0), 'timeout'
    if not q.empty():
        status, payload = q.get()
        if status == 'ok':
            return payload, 'ok'
        return dict(x=0.0, y=0.0, theta=0.0, scale=10.0, found=0, score=0.0), f'error: {payload}'
    return dict(x=0.0, y=0.0, theta=0.0, scale=10.0, found=0, score=0.0), 'unknown_failure'


def detect_columns(fieldnames, id_col, ref_col, search_col):
    def find(explicit, candidates):
        if explicit:
            return explicit
        for c in candidates:
            if c in fieldnames:
                return c
        return None
    id_c = find(id_col, ['pair_id', 'id', 'pairid'])
    ref_c = find(ref_col, ['ref_path', 'reference_path', 'ref_image', 'ref_image_path', 'reference', 'ref'])
    search_c = find(search_col, ['search_path', 'search_image', 'search_image_path', 'search'])
    return id_c, ref_c, search_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--id-col', default=None)
    ap.add_argument('--ref-col', default=None)
    ap.add_argument('--search-col', default=None)
    args = ap.parse_args()

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        id_c, ref_c, search_c = detect_columns(fieldnames, args.id_col, args.ref_col, args.search_col)
        print(f"pairs.csv columns: {fieldnames}")
        print(f"detected -> id_col={id_c!r} ref_col={ref_c!r} search_col={search_c!r}")
        if not (id_c and ref_c and search_c):
            print("ERROR: could not auto-detect columns. Re-run with --id-col/--ref-col/--search-col.")
            sys.exit(1)
        rows = list(reader)

    out_rows = []
    times = []
    for i, row in enumerate(rows):
        pair_id = row[id_c]
        ref_path = row[ref_c]
        search_path = row[search_c]
        t0 = time.time()
        r, status = run_one_pair(ref_path, search_path)
        dt = time.time() - t0
        times.append(dt)
        print(f"[{i+1}/{len(rows)}] {pair_id}: found={r['found']} score={r['score']:.3f} "
              f"time={dt:.2f}s status={status}")
        out_rows.append(dict(pair_id=pair_id, x=r['x'], y=r['y'], theta=r['theta'],
                              scale=r['scale'], found=r['found'], score=r['score']))

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pair_id', 'x', 'y', 'theta', 'scale', 'found', 'score'])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nwrote {len(out_rows)} rows to {args.output}")
    print(f"median wall-clock/pair: {np.median(times):.2f}s (budget: 5s target, 20s hard limit)")


if __name__ == '__main__':
    main()
