# Drift-Sense — Semicon India Hackathon Submission

AI-powered navigation-error recovery for wafer inspection tools.
Full proposal: `docs/proposal.pdf` (source: `docs/proposal.tex`).
Literature citations: `docs/citations.md`.

## Setup

    pip install -r requirements.txt

(The pinned versions in `requirements.txt` are a starting point — run
`pip freeze | grep -iE "numpy|scipy|pillow|torch"` on your own machine
after installing and use those exact versions for your own
reproducibility record, since they may differ slightly by platform.)

## Run order

    cd code
    python3 generate_dataset.py --out ./data --n 4 --seed 42
    python3 localize.py --reference data/pair_000_reference.png --search data/pair_000_search.png --zoom 10
    python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty easy
    python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty hard
    python3 evaluate.py --n 30 --tolerance 3.0 --seed 0 --difficulty hard --use_prior --max_drift_px 60
    python3 dl_localize.py --train --epochs 800 --size 300 --max_seconds 250 --out siamese_300.pt --seed 0
    python3 dl_localize.py --eval --weights siamese_300.pt --size 300 --n 30 --tolerance 3.0 --seed 0 --difficulty hard --refine --use_prior --max_drift_px 60

## Folder structure

    code/             all Python source (dataset generator, classical
                       localizer, DL ablation, batch evaluator)
    docs/              proposal.tex / proposal.pdf, citations.md
    sample_outputs/    pre-generated example images, eval JSON results,
                        and the trained DL weights (siamese_300.pt), so
                        results can be inspected without re-running
                        everything
    requirements.txt   Python dependencies

## Headline results

| Configuration | Success rate | Mean error | Compute time |
|---|---|---|---|
| Classical FFT-NCC, non-adversarial | 100% (30/30) | 0.03px | ~300ms |
| Classical FFT-NCC, adversarial decoy | 46.7% (14/30) | 195px | ~225ms |
| Classical FFT-NCC + prior window, adversarial decoy | **100% (30/30)** | **0.02px** | **~215ms** |
| DL (Siamese CNN) + refinement + prior | ~87-90% | ~5-8px | ~10-50ms |

Full methodology, literature justification, and honest failure-case
analysis are in `docs/proposal.pdf`.
