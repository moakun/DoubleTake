# Related work — living document (Gate 3, §0.C)

**Started:** 2026-07-23 · Maintained continuously; feeds the paper's related-work section.
**Novelty elements** (from §0.C of the plan):
- **E1** — tile/region selection for a second high-resolution pass
- **E2** — selection driven by a *learned predictor of expected detection gain* (value-of-information vs. a kept coarse pass)
- **E3** — *hard budget* top-K selection → deterministic per-frame latency, evaluated on accuracy vs. measured latency on embedded hardware
- **E4** — detector-agnostic wrapper around frozen weights, ≥3 detectors

**Kill criterion:** pivot only if a single prior work claims **E2+E3 together** for aerial/small-object detection.

---

## Clearance table

| Paper | Venue/Year | E1 | E2 | E3 | E4 | Delta (what they do / what we add) |
|---|---|---|---|---|---|---|
| **DenseScout** ([arXiv:2604.25300](https://arxiv.org/abs/2604.25300)) | arXiv 2026 | ✔ | ✗ | **✔** | ~ | **Closest prior work.** Learned dense selector (1.01M params), top-K under hard budget, RK3588 + Jetson Orin NX, VisDrone/DOTA/InsPLAD, introduces QoS-constrained recall. BUT: selector supervised by **GT center heatmaps (CenterNet focal loss)** = objectness/density, not gain; **no T1 coarse pass** — detections come only from selected patches, so unselected regions are blind. We add: gain labels vs. a kept T1 pass (incl. FP penalty), ranking losses, T1+T2 merge (full-frame coverage floor), 20× smaller head reusing T1 features, temporal amortization, Detection Latency metric. **Must cite and differentiate explicitly; validates feasibility of budgeted selection on Orin-class hardware.** |
| **Dynamic Zoom-in Network** (Gao et al.) ([arXiv:1711.05187](https://arxiv.org/abs/1711.05187), [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gao_Dynamic_Zoom-In_Network_CVPR_2018_paper.pdf)) | CVPR 2018 | ✔ | **✔** | ✗ | ✗ | **Closest on E2.** R-net predicts *accuracy gain* of high-res analysis per region; Q-net (RL) sequentially zooms with cost-aware reward. BUT: Caltech Pedestrians + YFCC (not aerial/small-object), sequential variable-latency zooms (no hard top-K budget, no deterministic latency), desktop evaluation, detector coupled to pipeline. We add: aerial small-object domain, deterministic budget, embedded measurement, supervised ranking instead of RL (simpler, deployable), frozen-detector wrapping. |
| **Remix** ([MobiCom 2021](https://dl.acm.org/doi/abs/10.1145/3447993.3483274), [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/08/Flexible-High-resolution-Object-Detection-on-Edge-Devices-with-Tunable-Latency.pdf)) | MobiCom 2021 | ✔ | ✗ | **✔** | ~ | Latency budget → non-uniform partition + per-block model plan on edge. Selection from offline profiling + historical object distribution (heuristic planner, surveillance-camera assumption of static viewpoint — breaks on a moving drone). No learned per-frame gain predictor. We add: learned per-frame VoI selection, moving-platform setting, frozen single detector. |
| **AdaZoom** ([arXiv:2106.10409](https://arxiv.org/abs/2106.10409)) | arXiv 2021 | ✔ | ~ | ✗ | ~ | RL focus-region generation for aerial (VisDrone/UAVDT/DOTA); reward from object distributions (density-flavored, not gain-over-coarse-pass). No latency budget, no embedded evaluation; collaborative training couples it to the detector. |
| **ClusDet / DMNet / GLSAN / UFPMP-Det family** | ICCV19/CVPRw20/…/AAAI22 | ✔ | ✗ | ✗ | ✗ | Density/cluster-guided crops for aerial images. Accuracy-motivated (crop → upsample → detect), typically *increases* compute; no budget, no gain labels, retrains detector components. |
| **CZDet** ([arXiv:2303.08747](https://arxiv.org/abs/2303.08747), [CVPRw 2023](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/supplemental/Meethal_Cascaded_Zoom-In_Detector_CVPRW_2023_supplemental.pdf)) | CVPRw 2023 | ✔ | ✗ | ✗ | ~ | Density crops as a learned pseudo-class inside the detector itself; second-stage inference on detected crops. Elegant, but density- not gain-driven, variable latency, detector retrained (crop class added). |
| **ESOD** ([arXiv:2407.16424](https://arxiv.org/html/2407.16424v1)) | 2024 | ✔ (feature-level) | ✗ | ✗ | ✗ | ObjSeeker + AdaSlicer skip background *feature computation* inside the network. Objectness-driven, architecture-integrated (not a wrapper), no hard budget. |
| **ASAHI** ([arXiv:2604.19233](https://arxiv.org/abs/2604.19233)) | 2026 | ✔ | ✗ | ✗ | ✔ | Adapts slice *count* (6/12) to image resolution — content-blind, no per-frame budget. Confirms the community sees uniform SAHI's waste; strengthens our motivation. Include as baseline variant if code available. |
| **SAHI** ([docs](https://docs.ultralytics.com/guides/sahi-tiled-inference)) | 2022 | ✔ (uniform) | ✗ | ✗ | ✔ | The baseline. Uniform slicing, detector-agnostic. Our primary competitor at equal latency. |
| **GigaDet** ([paper](https://ise.thss.tsinghua.edu.cn/mig/2022-2.pdf)) | 2021 | ✔ | ✗ | ~ | ✗ | Gigapixel video; patch-generation network locates object-likely areas + zoom ratios, real-time oriented. Objectness-driven, gigapixel domain, not budget-deterministic. |
| **SaccadeDet** ([arXiv:2407.17956](https://arxiv.org/abs/2407.17956)) | 2024 | ✔ | ✗ | ✗ | ? | Saccade (interest regions) + gaze (refine) on PANDA gigapixel + pathology WSI. Not aerial; interest ≠ gain. *Full method section not yet read — re-check if reviewers raise it.* |
| **QueryDet / CEASC** (from prior knowledge — **verify citations before writing**) | CVPR22 / AAAI23 | ✔ (feature-level) | ✗ | ✗ | ✗ | Sparse high-res heads / adaptive sparse conv where the *coarse features* fire — objectness-driven feature sparsity inside the detector. Orthogonal: could serve as a T2 detector under our scheduler. |
| Foveated/glimpse line (FOVEA ICCV21, SALISA ECCV22, glimpse nets) | — | ✔ (warp) | ~ | ✗ | ✗ | Magnify/warp by saliency; saliency ≠ measured detection gain; no hard budget; warping distorts geometry (hurts localization). Cite as the attention lineage. |

Legend: ✔ claims it · ~ partial · ✗ absent.

## Surviving delta (as of 2026-07-23)

No found work claims **E2+E3 together**, on aerial or otherwise: Gao et al. have E2 without E3 (and not aerial); DenseScout/Remix have E3 without E2. The conjunction — *gain-supervised, budget-deterministic, frozen-detector tile scheduling for UAV small objects* — appears unclaimed. Secondary reserves intact: nobody found uses (a) gain labels with an FP penalty against a kept T1 pass, (b) ranking objectives on measured tier-difference, (c) the Detection Latency metric, (d) staleness-based temporal amortization for detection scheduling.

**Sharpest positioning risk:** a reviewer collapses us into DenseScout ("learned budgeted patch selection on Jetson exists"). Pre-empt in the intro: coverage-supervised selection *is* the objectness baseline our Gate 2 quantifies headroom over; and patch-only pipelines have no coverage floor (T1 is kept in ours).

## Coverage gaps / TODO

- [ ] Forward-citation sweep of DenseScout and Gao et al. (Semantic Scholar) — highest-yield remaining check
- [ ] Read SaccadeDet + ASAHI full method sections
- [ ] Verify QueryDet/CEASC citations and claims from the actual papers
- [ ] ICRA/IROS 2024–2026 proceedings keyword pass ("budget", "anytime detection", "adaptive resolution UAV")
- [ ] VisDrone challenge workshop reports 2023–2025
- [ ] Set up monthly arXiv re-check (queries: "budgeted tile selection detection", "expected gain region selection", "value of information detection", "adaptive slicing UAV")
