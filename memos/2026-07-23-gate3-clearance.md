# Gate 3 clearance memo — prior-art sweep (§0.C)

**Date:** 2026-07-23 · **Status:** preliminary GO, first-pass sweep complete, four coverage TODOs open (see related_work.md)

## Call

**GO.** No prior work found claiming E2+E3 together (learned expected-gain selection + hard-budget deterministic latency), on aerial imagery or elsewhere. The two nearest works each hold exactly one of the two elements:

- **Dynamic Zoom-in Network (Gao et al., CVPR 2018)** holds E2: an R-net predicts the accuracy gain of high-resolution analysis per region. But: RL-sequential zooms with variable latency (no hard top-K budget), Caltech Pedestrians/YFCC (not aerial small-object), desktop-only, detector-coupled.
- **DenseScout (arXiv 2026)** holds E3: learned top-K patch selection under a hard budget on RK3588/Jetson Orin NX, evaluated on VisDrone/DOTA. But: the selector is supervised by GT center heatmaps (CenterNet focal loss) — objectness/density, not gain — and there is **no retained coarse detection pass**; unselected regions are entirely blind.

## Consequences for the paper

1. **DenseScout becomes the primary "vs." citation** and must be handled in the introduction, not related work. Framing: coverage-supervised selection is precisely the objectness baseline whose headroom Gate 2 measures; and a patch-only pipeline has no coverage floor, while DoubleTake keeps T1 everywhere and spends the budget only on *changing the answer*.
2. **Gao et al. must be credited for the gain-prediction idea.** Our claim narrows honestly to: first *budget-deterministic, embedded-measured, detector-agnostic* instantiation of gain-predicted second looks, in the aerial small-object domain, with supervised ranking replacing RL.
3. **DenseScout de-risks Gate 1**: budgeted selection on Orin-class hardware is publishable and latency evidently scales there — a good feasibility signal (still must be verified on our device/detector).
4. Secondary novelty reserves intact: FP-penalized gain labels vs. a kept T1 pass, ranking losses on measured tier-difference, Detection Latency metric, staleness-based temporal scheduling.

## Conditions on this GO

- Complete the four open coverage TODOs in related_work.md (forward-citations of DenseScout & Gao et al. are the highest-yield) **before the end of Phase 0**.
- Monthly arXiv re-checks for the project duration; DenseScout's existence (Apr 2026) shows this corner of the field is moving *now*.
- If a forward-citation of DenseScout turns up gain-supervised selection under budget, re-open this memo and re-decide.
