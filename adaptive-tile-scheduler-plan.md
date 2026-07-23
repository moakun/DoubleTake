# DoubleTake — Budget-Constrained Adaptive Resolution Scheduling for UAV Small-Object Detection

**Method name:** **DoubleTake** — a scheduler that learns where a second look is worth taking.

**Working title:** *DoubleTake: Where Is a Second Look Worth It? Budget-Constrained Tile Scheduling for Real-Time Small-Object Detection in UAV Imagery*

**Version:** 1.2 · Planning document · Not yet started
*(v1.1: added prior-art clearance protocol §0.C, Phase 0 detector-checkpoint note, tiered Phase 3 exit criterion, per-detector gain-label decision §2.4, isolated-vs-joint label validation §2.5, corrected budget relation to marginal cost, risks R9–R10. v1.2: named the method DoubleTake.)*

---

## 0. Thesis and scope

### 0.1 Claim to be defended

> Under a fixed inference latency budget, **selecting** which image regions receive a high-resolution second pass — using a learned predictor of *expected detection gain* rather than of objectness — dominates uniform tiling (SAHI-style) on the accuracy/latency Pareto frontier for UAV small-object detection, and does so detector-agnostically.

### 0.2 What this project is NOT

Explicitly out of scope, and stated in the paper to pre-empt reviewers:

- Not a new backbone, neck, attention module, or loss function.
- Not a claim of SOTA mAP on VisDrone at unconstrained latency.
- Not a super-resolution method. No pixels are hallucinated; only real sensor resolution is spent or not spent.

The contribution is an **inference-time scheduler** that wraps a frozen detector. This is deliberate: it survives the next YOLO/DETR release, which module-stacking papers do not.

### 0.3 Why the framing matters

The naive formulation — threshold the low-resolution objectness map, process hot tiles — fails for a structural reason:

> Objectness computed at low resolution is precisely the signal that is unreliable for small objects. If the coarse pass could see the 6-pixel pedestrians, the second pass would be unnecessary.

Therefore the scheduler must not predict *"where do I currently detect things"* but *"where would a second look change my answer"*. This is a **value-of-information** estimate. Reframing tile selection as expected-gain prediction is the intellectual core of the paper — and the source of the name: a *double take* is exactly a second look taken because the first glance may have gotten it wrong.

### 0.4 Notation (fix early, use consistently)

| Symbol | Meaning |
|---|---|
| `I` | Full-resolution input frame, `H × W` |
| `T1` | Tier-1 pass: whole frame downsampled to `s_lo × s_lo` |
| `T2` | Tier-2 pass: native-resolution tiles of size `s_t × s_t` |
| `G` | Tile grid, `g_h × g_w` candidate tiles with overlap `ρ` |
| `K` | Tile budget — number of tiles admitted to `T2` |
| `L_b` | Latency budget (ms), device-specific |
| `ĝ(t)` | Predicted gain for tile `t` |
| `g*(t)` | Oracle (measured) gain for tile `t` |
| `Π` | Scheduling policy mapping features → ordered tile list |

Budget relation: `K = ⌊(L_b − L_T1 − L_sched − L_merge − a_T2) / b_T2⌋`

where `a_T2` and `b_T2` are the fixed and marginal terms of the Gate 1 latency fit `L_T2(K) = a_T2 + b_T2·K`. The per-tile cost is the **marginal** cost `c_tile ≡ b_T2`; the fixed batch overhead `a_T2` (kernel launch, memcpy, batch setup) is paid once per frame, not per tile. Conflating the two overstates `c_tile` and silently under-fills the budget.

---

## Phase 0 — Feasibility gates (Weeks 1–2) · **KILL PHASE**

**Purpose:** Determine whether the paper can exist, before building anything. All three gates must pass (Gates 1 and 2 are empirical; Gate 3 is the literature clearance). If any fails, stop and revert to the alternative project. Do not write scheduler code during this phase.

**Detector used in Phase 0:** the frozen detector zoo does not exist until Phase 1, so both empirical gates run on an **off-the-shelf checkpoint** — a publicly available YOLOv11-s fine-tuned on VisDrone-train with a standard recipe (or the best available public VisDrone checkpoint). This is acceptable because the gates test properties of the *problem* (latency scaling of the hardware, spatial concentration of gain), not of a particular detector. **Re-verification rule:** once the Phase 1 zoo is frozen, re-run Gate 2's four-curve comparison with the primary detector (cheap — the tiling cache structure already exists). If the headroom criterion no longer holds with the real detector, treat it as a fresh Gate 2 failure and re-decide before Phase 2.

### 0.A Gate 1 — Latency must scale with tile count

**The risk being tested:** at small batch sizes on GPU, inference is often kernel-launch-bound rather than compute-bound. If batching 20 tiles costs approximately the same as batching 5, then *skipping tiles saves nothing* and the entire premise collapses.

**Procedure:**
1. Target device: NVIDIA Jetson Orin (Nano 8 GB and/or AGX), TensorRT FP16, MAXN power mode, fan fixed, thermals stabilized (10 min warm-up).
2. Export the frozen detector at tile input size `s_t ∈ {320, 512, 640}`.
3. Measure end-to-end latency for batch `K = 1 … 32`, 500 iterations each, discard first 50. Report **p50 and p95**, not mean.
4. Fit `L(K) = a + b·K`. Compute `b / a` — the marginal-to-fixed cost ratio.

**Pass criterion:** `b·K` accounts for **≥ 60%** of total latency at `K = 16`, and `L(K)` is monotonically increasing with an approximately linear regime across `K ∈ [4, 24]`.

**If it fails:**
- Move to a compute-bound regime: larger `s_t`, larger detector variant, or INT8→FP16.
- Or retarget the device: CPU (ARM Cortex on a Pi-class companion board) and edge NPUs show far cleaner scaling.
- If neither restores linearity, **kill the project**.

### 0.B Gate 2 — Oracle headroom must be large

**The risk being tested:** if tile selection barely matters, there is nothing to learn.

**Procedure:**
1. VisDrone2019-DET-val. Fix a tile grid, `ρ = 0.2` overlap.
2. For every image, run T2 on **every** tile exhaustively. Cache all per-tile detections.
3. Compute per-tile oracle gain `g*(t)` = number of ground-truth objects matched by T2-on-`t` that were **missed** by T1 alone (IoU ≥ 0.5, greedy matching, per-class).
4. Build four curves of AP_small vs. K:
   - **Oracle:** select top-K by `g*`
   - **Objectness:** select top-K by summed T1 objectness in tile region
   - **Uniform/SAHI:** first K tiles in raster order (equivalently, coarser uniform grid)
   - **Random:** K tiles sampled uniformly, averaged over 5 seeds

**Pass criteria (both required):**
- **Sparsity:** Oracle at `K = 0.25·|G|` recovers **≥ 90%** of the AP_small achieved by exhaustive tiling. *(Confirms the gain is concentrated in few tiles.)*
- **Learnability headroom:** Oracle at `K = 0.25·|G|` exceeds Objectness at the same K by **≥ 3.0 AP_small**. *(Confirms a learned predictor has something to beat.)*

**If Gate 2 fails:** the second criterion failing is fatal to the *learned* contribution but not necessarily to the paper — a well-engineered objectness-threshold system with a budget guarantee is still a workshop paper. Downgrade scope honestly rather than inflating a null result.

### 0.C Gate 3 — Prior-art clearance (Week 1, before any GPU time)

**The risk being tested:** R6 — someone has already published this. This is the cheapest fatal risk to retire; retire it first.

**What the paper actually claims, decomposed.** The novelty rests on the *conjunction* of four elements, not on any one of them:

1. **(E1)** Tile/region selection for a second high-resolution pass *(known: ClusDet, DMNet, CZDet, UFPMP-Det, AZNet, QueryDet all do coarse-to-fine region processing)*
2. **(E2)** Selection driven by a learned predictor of **expected detection gain** (value-of-information), not objectness/density
3. **(E3)** **Hard budget constraint** with top-K selection → deterministic per-frame latency, evaluated on the accuracy/*measured-latency* Pareto frontier on an embedded device
4. **(E4)** **Detector-agnostic wrapper** around frozen weights, demonstrated on ≥3 detectors

**Kill criterion, stated precisely:** the project pivots only if a single prior work claims **E2 + E3 together** for aerial/small-object detection. Prior work with E1 + objectness/density selection (the ClusDet/DMNet family) or E1 + E3 with a heuristic scorer narrows the delta but does not kill — it sharpens positioning. E4 and the Detection Latency metric (Phase 4) are secondary novelty reserves; note who is closest on each.

**Search protocol:**

- **Venues/indices:** Google Scholar, Semantic Scholar, arXiv (cs.CV), CVPR/ICCV/ECCV/WACV proceedings 2019–present, ICRA/IROS, ISPRS/TGRS; plus the VisDrone challenge reports (workshop papers often pre-empt ideas without being indexed well).
- **Query families** (run each, skim to saturation — no new relevant hits in 20 results):
  - *tiling/cropping:* "adaptive tiling detection", "learned tile selection", "slicing aided inference", "crop selection aerial detection", "coarse-to-fine UAV detection", "zoom-in detection"
  - *budget/latency:* "latency-constrained object detection", "anytime detection", "budgeted inference detection", "adaptive resolution inference"
  - *value-of-information:* "value of information visual attention", "expected gain region selection", "where to look detection", "glimpse network detection", "reinforcement learning region selection detection"
  - *adjacent mechanisms that could subsume the idea:* early-exit networks, dynamic neural networks (survey: Han et al.), saccade/foveated detection, active perception for UAVs
- **Forward-citation sweep:** follow citations *of* SAHI, ClusDet, DMNet, AZNet, and QueryDet — anyone doing learned budgeted selection will cite at least one of these.
- **Living document:** maintain `related_work.md` from week 1 with a table: paper / which of E1–E4 it claims / delta. This becomes the related-work section draft, so the effort is not overhead.

**Output:** a dated one-page clearance memo — closest prior work, the surviving delta, and an explicit go/pivot call. Re-run the arXiv queries monthly for the project's duration (a Scholar alert on the E2/E3 query families costs nothing); a mid-project scoop discovered early is a positioning problem, discovered at submission it is a rejection.

### Phase 0 deliverables

- `bench/latency_scaling.{py,csv}` + the `L(K)` plot
- `bench/oracle_headroom.{py,csv}` + the four-curve plot
- `related_work.md` + the one-page prior-art clearance memo (§0.C)
- **`GO/NO-GO memo` — one page, written and dated, covering all three gates.** The Gate 2 figure appears in the paper as Figure 2; it is the motivation, not just internal validation.

**Compute:** ~40 GPU-hours (exhaustive tiling of val set) + device access.

---

## Phase 1 — Infrastructure and honest baselines (Weeks 3–5)

The single most common failure in this subfield is comparing against an undertrained baseline. Everything below exists to make that impossible.

### 1.1 Frozen detector zoo

Train or obtain three detectors, **fully converged**, identical augmentation and schedule:

| Detector | Role | Notes |
|---|---|---|
| YOLOv11-s | Primary | Standard, comparable to related work |
| MEISCF (yours) | Continuity | Reuses your prior paper; shows the wrapper composes with domain-specific detectors |
| D-FINE-S or RF-DETR-N | Generalization | Proves the method is not YOLO-specific |

**Rule: detector weights are frozen for the entire project.** Every curve in every figure uses the same weights. If a reviewer can attribute a gain to detector retraining, the paper is dead.

Log the exact training recipe, seed, and final val AP for each. Publish these numbers so others can verify the baselines aren't sandbagged.

### 1.2 Latency harness

Non-negotiable requirements:
- Measured on the **target embedded device**, not a desktop GPU. Report the device.
- p50 and p95 latency, plus end-to-end (decode → merge), not model-only.
- Energy: mJ/frame via `tegrastats` sampling or an inline power monitor. **Almost nobody in this subfield reports energy — free differentiation.**
- Thermal protocol documented (sustained-load measurement, not cold-start bursts).

### 1.3 Baseline implementations

1. Full-frame at `s ∈ {512, 640, 960, 1280, 1536}` — the "just use bigger input" baseline, which is stronger than most papers admit.
2. Uniform SAHI at varying grid density and overlap — **tune this properly.** It is your real competitor.
3. Random-K and Objectness-K from Phase 0.
4. Oracle-K — upper bound, kept in every figure.

### Phase 1 deliverables

- Reproducible latency/energy harness (released)
- Frozen checkpoints + recipes for 3 detectors
- Baseline Pareto curves on VisDrone-val

**Exit criterion:** you can regenerate every baseline number with one command, on the device, with variance bars.

---

## Phase 2 — Gain-label construction (Weeks 6–7)

### 2.1 Label definition

For each training image and each candidate tile `t`, run T2 exhaustively and record:

- `g_tp(t)` — GT objects matched by T2-on-`t` but missed by T1 *(recovered detections)*
- `g_fp(t)` — false positives introduced by T2-on-`t` after merging *(the cost side)*
- `g_loc(t)` — sum of IoU improvement on objects detected by both tiers *(localization refinement)*

Composite target: `g*(t) = g_tp(t) + λ_loc · g_loc(t) − λ_fp · g_fp(t)`

Run a small sweep on `λ_fp ∈ {0, 0.25, 0.5, 1.0}` and report sensitivity. **Including the false-positive term is important** — high-resolution passes over empty terrain generate spurious detections, and a naive gain definition ignores that this actively hurts precision.

### 2.2 Boundary handling

Objects straddling tile edges are the main source of label noise.
- Overlap `ρ ≥ 0.2`, ablated at `{0.0, 0.1, 0.2, 0.3}`.
- Credit a recovered object to a tile only if **≥ 70%** of its GT box area lies inside that tile; otherwise credit both and mark the sample.
- Log the fraction of GT boxes affected — if it exceeds 15%, the grid is too fine.

### 2.3 Class imbalance in labels

Most tiles have `g* = 0`. Expect 75–90% zeros. Do not train a regressor on this naively; see 3.2.

### 2.4 Label-generation scope — decided up front

Gain labels are **detector-specific**: `g*(t)` is defined by what *this* detector's T1 pass missed. This quietly taxes the detector-agnosticism claim, so the policy is fixed now rather than discovered in Phase 5:

- **Labels are generated once, for the primary detector (YOLOv11-s) only.** The headline claim tested in §5.1 is that a scheduler *trained on one detector's gain labels* transfers zero-shot to the other two frozen detectors. This is the strong, interesting version of detector-agnosticism — and it is plausible because `g*` is dominated by *where small objects are* (a property of the scene) more than by detector idiosyncrasies.
- **Cheap early check (end of Phase 2):** before betting Phase 5 on it, generate labels for the second detector on a **10% subset** of VisDrone-train (~20 GPU-h) and report the Spearman rank correlation of per-image tile orderings between the two detectors' `g*`. If ρ ≥ 0.7, transfer is likely and the plan proceeds unchanged. If ρ < 0.7, exercise the contingency below *now*, not in week 18.
- **Contingency (pre-priced):** regenerate full labels for one additional detector (~200 GPU-h, already in the compute budget as a contingency line). If even per-detector labels do not rescue §5.1, the paper's claim downgrades from "detector-agnostic wrapper" to "wrapper with per-detector calibration" — weaker but survivable; §5.1 states which one honestly.

### 2.5 Label validity — isolated tiles vs. joint inference

Labels are computed by running T2 on each tile **in isolation** (merged against T1 only), but at inference the K selected tiles are merged **jointly**, and `g_fp` in particular can interact across overlapping tiles. Before trusting the labels:

- On a 500-image validation subset, compare the summed isolated gains `Σ g*(t)` of a top-K set against the **actually measured** AP change from jointly running and merging that same set, across `K ∈ {4, 8, 16}`.
- **Acceptance:** rank correlation between predicted-sum and measured gain ≥ 0.8 across image-K pairs. If it holds (expected — tile NMS in §3.3 suppresses the main interaction channel), state it in the paper as a one-line validation and move on. If it fails, add a joint correction: recompute `g_fp` with tile-pair co-occurrence on overlapping neighbors, and re-validate.
- Either way this pre-empts the reviewer objection that the training target mismatches the inference-time objective.

### Phase 2 deliverables

- Cached gain-label tensors for VisDrone-train (and UAVDT-train for Phase 5)
- Label statistics: gain distribution, zero fraction, boundary-affected fraction
- Cross-detector rank-correlation number from the §2.4 10% subset check, with the go/contingency call recorded
- Isolated-vs-joint validation result (§2.5)
- Dataset card documenting the generation procedure

**Compute:** ~150–250 GPU-hours (exhaustive tiling of the training set). This is the largest single compute item. Cache aggressively; regenerate nothing.

---

## Phase 3 — The scheduler (Weeks 8–12) · **CORE CONTRIBUTION**

### 3.1 Architecture constraints

The scheduler must be nearly free or the entire premise self-destructs.

- **Reuses T1 features.** No second backbone. Tap one pyramid level (ablate which).
- **Parameter budget: < 50k.** Report exact params and FLOPs.
- **Overhead budget: `L_sched` < 10% of `c_tile`.** State it in the abstract.
- Output: dense gain map at grid resolution, `g_h × g_w`.

Sketch (3–4 conv layers, depthwise-separable, single output channel). If it needs more than that, the features are wrong — fix the tap point, not the head size.

### 3.2 Training objective — ranking, not regression

**This is a design decision worth a paragraph in the paper.** Top-K selection depends only on the *ordering* of tiles, not on calibrated gain magnitudes. An L2 regressor spends capacity fitting the magnitude of the largest gains and is dominated by the zero-mass.

Compare, as an ablation:
1. L2 regression on `g*` — baseline objective
2. Focal/weighted regression accounting for zero-inflation
3. **Pairwise ranking loss** (RankNet-style) on tiles within an image
4. **Listwise loss** (ListNet / soft-top-K, e.g. differentiable relaxation via `SoftSort`)

**Prediction to state up front, so it is falsifiable:** listwise ≥ pairwise > weighted regression > L2. If the ordering comes out otherwise, that result is itself publishable and should be reported as-is.

### 3.3 Budget-constrained selection

- Fixed `K` from `L_b` → **deterministic latency**, which is the whole point for a real-time control loop. A confidence threshold gives variable latency and is unacceptable on a drone; say this explicitly.
- Top-K via a single vectorized `topk` kernel. Never a Python loop.
- Optional: non-maximum tile suppression — penalize selecting heavily overlapping tiles so the budget is spent on distinct regions. Ablate on/off.

### 3.4 Merging

- Batch **all** K tiles into one forward pass. Never call the model K times.
- Resolution-aware conflict resolution: when both tiers detect the same object, **the high-resolution box wins outright.** Do not average boxes; do not let confidence-ranked NMS arbitrate — the low-resolution box is confidently wrong about extent. Ablate this against plain IoU-NMS; expect ~1 AP_small.
- Coordinate transforms must be exact. Off-by-one tile origins produce a silent, uniform AP penalty that is very hard to debug later. Write a unit test with synthetic boxes.

### Phase 3 deliverables

- Trained scheduler; primary Pareto figure (yours vs. SAHI vs. full-frame vs. oracle)
- Loss-function ablation table
- Overhead accounting table: `L_T1`, `L_sched`, `c_tile`, `L_merge`

**Exit criterion — tiered, decided in advance so the outcome doesn't get rationalized after the fact.** All tiers measured at `K` corresponding to a 33 ms budget, vs. *tuned* uniform SAHI at equal measured latency, over 3 seeds:

| Outcome | Call |
|---|---|
| **≥ 2.0 AP_small**, gap significant over 3 seeds | Full paper as planned; proceed to Phase 4. |
| **1.0 – 2.0 AP_small**, significant | Paper survives but reframes: lead with **deterministic latency + energy + tight-budget regime** as the contribution, accuracy gain as secondary. Skip or compress Phase 4 to protect the timeline; target WACV applications track or ICRA over a main CV venue. |
| **< 1.0 AP_small**, or gap within seed variance | The learned scheduler does not beat its baseline meaningfully. Two honest exits: (a) if the *objectness*-K baseline with budget guarantee still beats SAHI, publish the engineered system as a workshop/short paper; (b) otherwise this is Gate 2's promise failing to materialize — write the negative-result memo, stop, and salvage the released harness + gain labels as a standalone dataset/benchmark contribution. |

**Expected honest outcome:** you win at tight budgets and converge with SAHI at loose budgets. That is a good result. Report the crossover point explicitly — it pre-empts the obvious reviewer objection and makes the paper more credible, not less.

---

## Phase 4 — Temporal amortization (Weeks 13–16) · **STRENGTH MULTIPLIER**

This is what elevates the paper from "a good systems trick" to a distinctive contribution: the second look becomes nearly free because its cost is amortized across frames.

### 4.1 Setup

- VisDrone2019-**VID** (video), plus UAVDT sequences.
- Scheduler input augmented with prior-frame state: previous detections warped by ego-motion, and a per-tile staleness counter (frames since last T2 visit).

### 4.2 Ego-motion compensation

Tiles must be tracked in world coordinates, not image coordinates, or a moving drone invalidates all priors.
- Sparse optical flow / homography estimation between consecutive frames (cheap: ~2–3 ms with a downsampled Lucas-Kanade or ECC on the T1 image).
- Ablate: with vs. without compensation. **Prediction: without it, temporal priors help at hover and actively hurt during fast translation.** Measure both regimes separately — split sequences by estimated ego-motion magnitude.

### 4.3 The exploration problem

A purely exploitative policy revisits known-hot tiles and can leave a newly-entering object undetected for hundreds of frames. This is a **safety-relevant failure mode**, not a minor accuracy issue, and it deserves its own metric.

Policies to compare:
1. Greedy exploit — top-K by predicted gain
2. ε-greedy — fraction of budget on random cold tiles
3. **Staleness-weighted** — `score = ĝ(t) + β · staleness(t)`, i.e. UCB-flavored
4. Coverage-guaranteed — hard constraint that every tile is visited within `W` frames

**New metric to introduce: Detection Latency** — frames between an object's first appearance in-frame and its first detection. Report the distribution (median and p95), not just the mean. This metric does not exist in the UAV detection literature and is arguably more operationally meaningful than AP for a tracking/avoidance pipeline. **This may end up being the most-cited idea in the paper.**

### Phase 4 deliverables

- Temporal scheduler + exploration-policy comparison
- Detection Latency distributions per policy
- Accuracy vs. amortized latency curve on video
- Ego-motion regime breakdown

---

## Phase 5 — Generalization (Weeks 17–19)

Answering the two questions any competent reviewer will ask.

### 5.1 Detector-agnosticism

Full Pareto curves for all three frozen detectors from Phase 1, using the **single scheduler trained on YOLOv11-s gain labels** (the zero-shot transfer claim fixed in §2.4). The §2.4 rank-correlation check will already have predicted whether this works; if the contingency was exercised, report both the zero-shot and per-detector-label variants and state plainly which claim the evidence supports. **If the wrapper only works with your own detector, the paper's central selling point is false** — report that honestly if so, and reframe per the §2.4 downgrade path.

### 5.2 Cross-domain transfer

Train the scheduler on VisDrone, evaluate zero-shot on:
- UAVDT (different capture conditions, altitudes)
- AI-TOD (aerial, tinier objects, different sensor)
- HIT-UAV (thermal — hardest transfer, and the most interesting result either way)

Compare: zero-shot scheduler vs. fine-tuned scheduler vs. objectness baseline.

**Rationale:** almost every X-YOLO paper in this subfield trains and tests exclusively on VisDrone and shows no transfer evidence at all. Producing that evidence is cheap for you (the scheduler is tiny and the detector stays frozen) and differentiates the work immediately.

---

## Phase 6 — Deployment reality check (Weeks 20–21)

- Full pipeline on-device, TensorRT, end-to-end including image decode and merge.
- Verify **deterministic latency**: p95/p50 ratio should be near 1.0. Report it — this is the practical claim that a threshold-based system cannot make.
- Energy per frame across the budget sweep.
- Sustained-load thermal behavior over 10+ minutes.
- Failure gallery: dense urban scenes, near-empty scenes, transition frames, motion blur. Include the ugly cases in the paper.

---

## Phase 7 — Ablations, writing, submission (Weeks 22–26)

### 7.1 Required ablation table

| Component | Variants |
|---|---|
| Gain target | `g_tp` only / `+ g_loc` / `+ g_fp` penalty; `λ` sweep |
| Loss | L2 / weighted / pairwise / listwise |
| Feature tap level | P2 / P3 / P4 |
| Tile overlap `ρ` | 0.0 / 0.1 / 0.2 / 0.3 |
| Tile size `s_t` | 320 / 512 / 640 |
| Merge policy | resolution-aware / plain NMS / WBF |
| Tile NMS | on / off |
| Temporal prior | none / naive / ego-compensated |
| Exploration | greedy / ε-greedy / staleness / coverage |

**Every number: 3 seeds, mean ± std.** Given how much of this subfield's reported gains plausibly sit inside seed variance, reporting variance is itself a differentiator — and it protects you when someone eventually audits these papers.

### 7.2 Figure plan

1. Motivation — heatmap of GT object density showing spatial sparsity across a dataset
2. **Phase 0 oracle headroom** — the go/no-go figure
3. Architecture diagram
4. **Main Pareto: AP_small vs. measured on-device latency** (the money figure)
5. Pareto vs. energy (mJ/frame)
6. Detection Latency distributions (temporal)
7. Cross-detector Pareto (3 panels)
8. Qualitative: selected tiles overlaid on frames, including failures

### 7.3 Positioning against related work

Be explicit and generous:
- **vs. SAHI / uniform tiling:** same mechanism, learned allocation, budget guarantee
- **vs. adaptive-resolution / dynamic-inference work** (glimpse networks, adaptive-focus, region-proposal-then-zoom): differentiate on *budget-constrained* selection and *detector-agnostic* wrapping
- **vs. the X-YOLO literature** (MEIS-YOLO, CF-YOLO, FCA-YOLO, EMFE-YOLO, SFFEF-YOLO, …): orthogonal, composable — position as complementary, not competing. **These are your reviewers. Cite them properly and show your method improves them.**

The prior-art sweep is Gate 3 (§0.C) and happens in week 1, not Phase 7 — by now `related_work.md` is a living document and this section is largely assembled from it. The monthly arXiv re-checks from §0.C mean no scoop should surface for the first time at writing.

### 7.4 Venue

| Target | Fit |
|---|---|
| ICRA / IROS | Strong — embedded, deployment, energy, latency guarantees are native concerns |
| WACV | Strong — applications track rewards systems contributions |
| ISPRS J. Photogramm. Remote Sens. | Strong journal option, high impact factor, remote-sensing audience |
| IEEE TGRS / RSE | Good, slower |
| Remote Sensing (MDPI) / Sci Rep | Fallback, faster, lower ceiling |

Recommendation: aim conference-first (ICRA/WACV) for the Phase 0–3 core, then extend with Phases 4–6 into a journal version. The temporal work is a natural journal extension.

---

## Timeline summary

| Phase | Weeks | Output | Kill gate |
|---|---|---|---|
| 0 · Feasibility | 1–2 | Go/no-go memo (Gates 1–3) | **YES** |
| 1 · Infra + baselines | 3–5 | Harness, frozen zoo, baselines, Gate 2 re-check | — |
| 2 · Gain labels | 6–7 | Cached labels + stats, §2.4/§2.5 validations | Soft |
| 3 · Scheduler | 8–12 | Core result, main Pareto | **YES** |
| 4 · Temporal | 13–16 | Detection Latency, policies | Soft |
| 5 · Generalization | 17–19 | Cross-detector, cross-domain | — |
| 6 · Deployment | 20–21 | On-device, energy, thermal | — |
| 7 · Ablations + writing | 22–26 | Submission | — |

**~6 months to conference submission.** Phase 0 alone is 2 weeks and answers whether the remaining 24 are worth spending. Do not skip it, and do not start Phase 3 code before Phase 0 passes.

---

## Compute budget estimate

| Item | GPU-hours |
|---|---|
| Detector training/verification (3 models) | 120 |
| Phase 0 oracle (val only) | 40 |
| Phase 2 gain labels (train sets, primary detector — §2.4) | 200 |
| §2.4 cross-detector 10% subset check + §2.5 joint validation | 30 |
| Scheduler training (small, many runs) | 80 |
| Ablations, 3 seeds | 150 |
| Cross-domain | 60 |
| **Planned total** | **~680 GPU-hours** |
| Contingency: full labels for a second detector (§2.4, only if ρ < 0.7) | +200 |
| **Worst case** | **~880 GPU-hours** |

Plus dedicated embedded-device access for Phases 0, 1, 6. **The device is on the critical path — secure it before Phase 0**, not before Phase 6.

---

## Consolidated risk register

| # | Risk | Detection | Mitigation | Severity |
|---|---|---|---|---|
| R1 | Latency doesn't scale with K | Phase 0 Gate 1 | Change regime or device; else kill | **Fatal** |
| R2 | No oracle headroom over objectness | Phase 0 Gate 2 | Downgrade to engineered system | **Fatal** |
| R3 | Scheduler overhead eats the savings | Phase 3 accounting | < 50k params, reuse T1 features | High |
| R4 | Boundary objects dominate label noise | Phase 2 stats | Overlap + area-based credit rule | Medium |
| R5 | Tuned SAHI is stronger than expected | Phase 1 baselines | Report honestly; emphasize tight-budget regime and determinism | Medium |
| R6 | Prior art already exists | §0.C protocol, week 1 + monthly re-checks | Precise kill criterion (E2+E3 conjunction); pivot early | High |
| R7 | Temporal priors fail under ego-motion | Phase 4 regime split | Ego-compensation; report both regimes | Low |
| R8 | Gains within seed variance | 3-seed protocol throughout | Variance bars from the start | Medium |
| R9 | Gain labels don't transfer across detectors | §2.4 10% rank-correlation check, end of Phase 2 | Pre-priced +200 GPU-h contingency; downgrade claim per §2.4 | Medium |
| R10 | Isolated-tile labels mismatch joint inference | §2.5 validation, end of Phase 2 | Tile NMS; joint `g_fp` correction if rank corr < 0.8 | Low |

---

## Standing principles

1. **Frozen detector weights, always.** One retrained baseline destroys the paper's attribution.
2. **Measured latency on the target device.** Desktop FPS numbers are meaningless for the stated application.
3. **Oracle curve in every accuracy figure.** It shows headroom and signals honesty.
4. **Three seeds, variance bars, everywhere.** Non-negotiable.
5. **Report the crossover point where SAHI catches up.** Stating your method's limits makes the rest credible.
6. **Release the harness and the gain labels.** The labels are reusable by others; that drives citations more reliably than the method itself.
