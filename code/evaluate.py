"""
evaluate.py
===========
Batch evaluation harness for the Drift-Sense localizer.

Runs the pipeline (generate_dataset.generate_pair -> localize.localize)
across N randomized synthetic pairs and reports, per the hackathon
scoring rubric:

  - computation time per single 1000x1000 image
  - % of cases landing within a stated tolerance of true location
  - a running/live counter after every case, so you can literally watch
    the success rate and average speed converge as more cases run --
    this is what you'd screen-record or paste into your presentation to
    show the algorithm's progress/efficiency case by case, not just a
    single final number.
  - at least one concrete honest failure example, with its PSR/ambiguity
    diagnostics, for the explainability portion of scoring.

USAGE
-----
    python evaluate.py --n 30 --tolerance 3.0 --seed 0
"""

import argparse
import json
import time

import numpy as np

from generate_dataset import generate_pair, to_uint8
from localize import localize


def run_case(seed, tolerance_px, zoom_ratio=10, structure="dram", difficulty="easy",
             use_prior=False, drift_std_px=15.0, max_drift_px=60.0, prior_rng=None):
    ref, search, gt = generate_pair(size=1000, seed=seed, zoom_ratio=zoom_ratio,
                                     structure=structure, difficulty=difficulty)
    t0 = time.time()

    expected_xy = None
    if use_prior:
        # Simulate the commanded/expected coordinate: the true location
        # plus a random bounded drift error (this is what "navigation
        # error recovery" actually means -- the tool KNOWS where it told
        # the stage to go; it doesn't know the exact drift).
        pr = prior_rng or np.random.default_rng(seed + 777)
        expected_xy = (gt["x"] + pr.normal(0, drift_std_px),
                        gt["y"] + pr.normal(0, drift_std_px))

    result = localize(ref, search, zoom_ratio=zoom_ratio,
                       expected_xy=expected_xy,
                       max_drift_px=max_drift_px if use_prior else None)
    elapsed = time.time() - t0  # wall-clock, independent of localize()'s own timer

    c = result["chosen"]
    err = float(np.hypot(c["x"] - gt["x"], c["y"] - gt["y"]))
    success = err <= tolerance_px

    return {
        "seed": seed,
        "true_x": gt["x"], "true_y": gt["y"],
        "pred_x": c["x"], "pred_y": c["y"],
        "error_px": err,
        "success": success,
        "psr": c["psr"],
        "global_ambiguity_ratio": result["global_ambiguity_ratio"],
        "ambiguous_flagged": result["ambiguous"],
        "compute_time_ms": elapsed * 1000.0,
        "n_candidates_considered": len(result["candidates"]),
    }


def print_progress_row(i, n, running, tolerance_px):
    """Live 'efficiency/progress' counter printed after every case:
    cumulative success rate and cumulative average speed so far."""
    successes = sum(r["success"] for r in running)
    avg_err = np.mean([r["error_px"] for r in running])
    avg_time = np.mean([r["compute_time_ms"] for r in running])
    success_rate = 100.0 * successes / len(running)
    bar_len = 30
    filled = int(bar_len * len(running) / n)
    bar = "#" * filled + "-" * (bar_len - filled)

    last = running[-1]
    mark = "OK " if last["success"] else "MISS"
    print(
        f"[{bar}] {i:>2}/{n}  "
        f"case={mark}  err={last['error_px']:6.2f}px  t={last['compute_time_ms']:6.1f}ms  "
        f"gar={last['global_ambiguity_ratio']:.3f}  "
        f"|  running: success={success_rate:5.1f}% ({successes}/{len(running)})  "
        f"avg_err={avg_err:6.2f}px  avg_time={avg_time:6.1f}ms"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="number of randomized test cases")
    ap.add_argument("--tolerance", type=float, default=3.0,
                     help="success tolerance in search-image pixels (10 nm/px)")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--structure", choices=["dram", "finfet"], default="dram")
    ap.add_argument("--difficulty", choices=["easy", "hard"], default="easy")
    ap.add_argument("--use_prior", action="store_true",
                     help="restrict search to a window around a simulated "
                          "commanded/expected coordinate (bounded drift prior)")
    ap.add_argument("--drift_std_px", type=float, default=15.0)
    ap.add_argument("--max_drift_px", type=float, default=60.0)
    ap.add_argument("--out", default="./eval_results.json")
    args = ap.parse_args()

    print(f"Running {args.n} randomized '{args.structure}'/{args.difficulty} test cases "
          f"(tolerance = {args.tolerance:.1f}px = {args.tolerance*10:.0f} nm)"
          f"{'  [prior window: max_drift=%dpx]' % args.max_drift_px if args.use_prior else ''}\n")

    running = []
    t_start = time.time()
    for i in range(1, args.n + 1):
        r = run_case(seed=args.seed + i, tolerance_px=args.tolerance,
                      structure=args.structure, difficulty=args.difficulty,
                      use_prior=args.use_prior, drift_std_px=args.drift_std_px,
                      max_drift_px=args.max_drift_px)
        running.append(r)
        print_progress_row(i, args.n, running, args.tolerance)
    total_time = time.time() - t_start

    successes = sum(r["success"] for r in running)
    errors = [r["error_px"] for r in running]
    times = [r["compute_time_ms"] for r in running]

    summary = {
        "n_cases": args.n,
        "tolerance_px": args.tolerance,
        "success_rate_pct": 100.0 * successes / args.n,
        "n_success": successes,
        "n_fail": args.n - successes,
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "max_error_px": float(np.max(errors)),
        "mean_compute_time_ms": float(np.mean(times)),
        "p95_compute_time_ms": float(np.percentile(times, 95)),
        "total_wall_time_s": total_time,
        "cases": running,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success rate       : {summary['success_rate_pct']:.1f}% "
          f"({successes}/{args.n} within {args.tolerance:.1f}px)")
    print(f"Mean / median error: {summary['mean_error_px']:.2f}px / "
          f"{summary['median_error_px']:.2f}px")
    print(f"Max error          : {summary['max_error_px']:.2f}px "
          f"(worst case -> inspect for the failure writeup)")
    print(f"Mean compute time  : {summary['mean_compute_time_ms']:.1f}ms per 1000x1000 image")
    print(f"p95 compute time   : {summary['p95_compute_time_ms']:.1f}ms")

    failures = [r for r in running if not r["success"]]
    if failures:
        worst = max(failures, key=lambda r: r["error_px"])
        print(f"\nWorst failure: seed={worst['seed']}  error={worst['error_px']:.2f}px  "
              f"psr={worst['psr']:.2f}  ambiguous_flagged={worst['ambiguous_flagged']}")
        print("-> Re-run this seed through generate_dataset.py + localize.py directly "
              "to produce the annotated failure figure for your write-up.")
    else:
        print("\nNo failures at this tolerance -- consider tightening --tolerance "
              "or generating a deliberately harder/more-periodic test case "
              "(see the ambiguous-case generator, next step) to get an honest "
              "failure example for the explainability section.")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
