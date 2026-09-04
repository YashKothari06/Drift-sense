# Citations — Drift-Sense

Every design choice below is paraphrased from real, verifiable sources (not
quoted) and mapped directly to where it's used in the code, so this can
drop straight into your presentation's "justify every augmentation choice"
requirement.

## 1. SEM noise model (Poisson-Gaussian) — `apply_sem_noise()` in generate_dataset.py

SEM images are widely characterized as having a signal-dependent Poisson
(shot-noise) component combined with a signal-independent Gaussian
(read/amplifier-noise) component — a "mixed Poisson-Gaussian" model. This
is stated across multiple independent sources on electron/optical
microscopy noise, not just one lab's convention:

- Avci et al., "Poisson shot noise parameter estimation from a single
  scanning electron microscopy image" — explicitly notes SEM noise stems
  from a Poisson process depending on signal level, not pure Gaussian
  noise as sometimes assumed.
  https://www.researchgate.net/publication/258813850
- "M-Denoiser: Unsupervised image denoising for real-world optical and
  electron microscopy data" (ScienceDirect, 2023) — describes microscopy
  noise as composed of signal-dependent shot noise plus signal-independent
  read noise, modeled as Poisson-Gaussian, and notes it is spatially
  correlated due to the acquisition process (motivates our scan-line
  jitter term too).
  https://www.sciencedirect.com/science/article/abs/pii/S0010482523007734
- Oxford Academic, "Scanning Electron Microscopy Noise Classification
  Using Machine Learning" (2025) — confirms Gaussian/Poisson noise
  attributable to electron source, beam-specimen interaction, and
  detectors; notes prior deep-learning work trains on synthetic
  Gaussian+Poisson-corrupted SEM images, same approach used here.
  https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.236
- Prasad & Joy (2003), discussed in "Is SEM Noise Gaussian?" — the SEM's
  noise character (Gaussian, Poisson, or mixed) depends on dwell
  time/scanning speed, motivating why we made the wide-search image
  noisier than the reference (implies shorter effective dwell time on a
  fast survey scan vs. a careful characterization scan).
  https://www.researchgate.net/publication/332907246

## 2. Line-edge roughness (LER/LWR) — `_smooth_noise_1d()` / edge jitter in `render_dram_pattern`, `render_finfet_pattern`

Real lithographic/etched edges are not geometrically straight; edge
position varies stochastically along the line, characterized in the
industry as LER (line-edge roughness) / LWR (line-width roughness):

- Bunday, Bishop, Villarrubia, Vlädar (NIST/International SEMATECH AMAG),
  "CD-SEM Measurement of Line Edge Roughness Test Patterns for 193 nm
  Lithography" — the foundational LER metrology reference; notes LER
  became a major industry concern per the 2001 ITRS roadmap.
  https://www.nist.gov/publications/cd-sem-measurement-line-edge-roughness-test-patterns-193-nm-lithography
- "Influence of image processing on line-edge roughness in CD-SEM
  measurement" — gives concrete target numbers: ITRS calls for line
  roughness below ~1.7 nm for sub-20nm nodes, though real lithography at
  the time achieved only ~4-5 nm — useful for justifying a specific
  ler_amplitude_px value in your writeup rather than an arbitrary one.
  https://www.researchgate.net/publication/253321431
- Science.gov summary on LWR — notes CD-SEM measurement itself has
  significant high-frequency random noise limiting roughness measurement
  resolution, which is part of why we treat LER as a *smoothed* (not
  white) random process (short correlation length, not per-pixel iid).
  https://www.science.gov/topicpages/l/line-width+roughness+lwr.html

## 3. FinFET fin/gate pitch parameters — `render_finfet_pattern()`

- "Scaling of SOI FinFETs down to fin width of 4 nm for the 10 nm
  technology node" (IEEE) — reports a fabricated FinFET array with fin
  pitch (FP) = 40 nm, fin width (DFin) = 4 nm, gate length (LG) = 20 nm.
  These are realistic, citable numbers for justifying pitch/width
  parameter choices in the FinFET generator mode.
  https://ieeexplore.ieee.org/document/5984609/
- SemiEngineering, "Re-Engineering The FinFET" — describes fin pitch as
  fin width + inter-fin spacing, and notes the industry targets ~0.7x
  pitch scaling per node, useful context for why pitch parameters should
  be presented as a swept/parametrized range rather than one fixed value.
  https://semiengineering.com/re-engineering-the-finfet/

## 4. Optical-microscope bonus: chromatic aberration + diffraction PSF — `to_rgb_optical()` in generate_dataset.py

- Nikon MicroscopyU, "The Diffraction Barrier in Optical Microscopy" —
  describes the Airy-disk diffraction pattern and typical lateral
  resolution limits (~200-250 nm) even for the best objectives, and notes
  the recorded image is the object convolved with the point-spread
  function — the basis for our per-channel Gaussian PSF blur.
  https://www.microscopyu.com/techniques/super-resolution/the-diffraction-barrier-in-optical-microscopy
- Wadsworth Center (NYS DOH) microscopy glossary — defines chromatic
  aberration precisely as shorter wavelengths refracting more than longer
  ones at a lens surface, causing the focal length (and thus effective
  magnification/registration) to differ slightly per color channel — the
  direct justification for applying an independent sub-pixel shift per
  R/G/B channel rather than a single shared blur.
  https://www.wadsworth.org/research/cores/alm/glossary

## 5. Fast NCC / matched filtering — `normalized_cross_correlation_fft()` in localize.py

- Lewis, J.P., "Fast Normalized Cross-Correlation," Vision Interface,
  1995 (also released as an Industrial Light & Magic technical report) —
  the standard reference for computing NCC efficiently via FFT for the
  numerator and running-sum ("integral image") tables for the local
  variance in the denominator, exactly the method implemented here.
  Confirmed independently by multiple later papers restating the same
  algorithm and complexity (O(N log N), FFT-dominated).
  https://www.researchgate.net/publication/228940930 (algorithm summary)
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203434
  (independent confirmation of Lewis's approach and its FFT-domain cost)

## 6. Peak-to-Sidelobe Ratio (PSR) — `peak_to_sidelobe_ratio()` in localize.py

- Bolme, Beveridge, Draper, Lui, "Visual Object Tracking Using Adaptive
  Correlation Filters" (MOSSE tracker), CVPR 2010 — introduces PSR =
  (peak - sidelobe_mean) / sidelobe_std as the confidence metric for a
  correlation-filter tracker, used to detect occlusion/tracking failure.
  This is the exact formula and source domain (correlation-filter
  tracking, not fab metrology) we're deliberately borrowing here — worth
  stating explicitly in your presentation as the source of the "novel
  angle," including its documented failure mode we found empirically
  (local PSR doesn't see a distant decoy — see `global_ambiguity_ratio`
  in localize.py for the metric that does).
  https://ieeexplore.ieee.org/abstract/document/5539960
  https://impact.ornl.gov/en/publications/visual-object-tracking-using-adaptive-correlation-filters/

## 7. Prior-window search restriction ("navigation-error recovery" framing) — `expected_xy`/`max_drift_px` in `localize()`

- US Patent 7,545,497, "Alignment routine for optically based tools" —
  describes defining a point of interest, then scanning a LOCAL area near
  that point (using a known periodicity) to find a matching unique
  feature, rather than searching the full field blind. This is the exact
  mechanism our prior-window restriction implements, and it's the reason
  we found a genuine no-trade-off fix (100% success, same speed) rather
  than just a better disambiguation heuristic on the same blind search.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7545497
- "Coarse-to-fine search method, image processing device and recording
  medium" (patent) and Zhang et al., "Coarse-to-Fine Object Tracking Using
  Deep Features and Correlation Filters" (arXiv 2012.12784) — both
  independently describe the same general principle in different domains
  (image search acceleration; visual object tracking): narrow the search
  region using a coarse/prior estimate before running the expensive fine
  match, which is both faster AND more robust to look-alike distractors
  outside the window.
  https://arxiv.org/pdf/2012.12784
- Die bonding apparatus patent (12327739) — a semiconductor-specific
  example of the same idea: rough-position a die first (via dicing-groove
  detection), THEN run template matching only within a small search area
  around that rough position for fine positioning.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12327739

---

## How to use this in your submission "every augmentation/noise choice... against
at least 2-3 credible public sources." The mapping above already groups
sources by exactly the noise/augmentation categories the rubric names
(noise model, LER/roughness, structural parameters, optical bonus). For
your slide, a table with columns [Design choice | Parameter value used |
Source(s)] built directly from the sections above will satisfy this
requirement directly — you don't need additional citation hunting unless
you change the underlying design choices.
