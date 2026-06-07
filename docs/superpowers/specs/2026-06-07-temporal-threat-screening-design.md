# Design Spec: Low-Label Temporal Threat Screening

**Date:** 2026-06-07
**Status:** Approved (brainstorming complete)
**Type:** Portfolio use-case extension of the low-label-defect-detection repo

## 1. Context & motivation

This repo today is a portfolio project on label-efficient image learning: SimCLR
self-supervised pretraining + hard pseudo-labeling, demonstrated on MNIST and MVTec AD.
The goal of this work is to extend it into a **product-flavored use-case with a live demo**:
airport-style **X-ray baggage threat screening** that accounts for **time/movement** — a bag
passes a scanner over roughly 5 seconds — and ships with a demo a person can watch and trust.

The intended outcome is a **portfolio showcase**: it must look and feel real and be
technically honest, but does not need certified production accuracy.

The honest bridge between "what the repo already does well" and "security screening":
**labeled threat data — especially bounding boxes — is scarce, expensive, and sensitive.**
That is exactly the regime where self-supervised + semi-supervised label-efficiency wins.
This is literature-backed: point-supervised and weakly-semi-supervised X-ray detection with
**pseudo bounding boxes** is an active 2024–2026 research line (I²OL-Net, BCR-Net, Mix-Paste).

## 2. Locked decisions

1. **Goal:** portfolio showcase — honest, impressive, not production-certified.
2. **Temporal role:** multi-frame **evidence accumulation under a confidence gate**. As the bag
   passes, per-frame predictions are aggregated for each tracked object; the system emits
   **AUTO-CLEAR / AUTO-FLAG / → REVIEW QUEUE**. (Not "motion is the signal" — there is no
   public temporal X-ray video, so that interpretation would require fabricating the very data
   the claim rests on.)
3. **Demo sensor:** **both** — (a) synthesized X-ray pass-through (pan a scan window across a
   real still X-ray over ~5s, mimicking line-scan), and (b) live RGB webcam. They share the
   temporal engine; only the frame source differs.
4. **Compute:** train on **free Colab/Kaggle T4**, download the checkpoint, run the **demo
   locally on CPU/MPS**.
5. **Task framing:** binary threat / no-threat **headline** + multi-class **bonus** figure.
6. **Model direction:** **detection-centric** (bounding boxes), phased crawl→walk→run.
   - **Hero:** **RF-DETR** (Roboflow, ICLR 2026) — DINOv2 self-supervised backbone, ~30M params
     (Nano/Small), fine-tunable on a free Colab T4 with `batch_size=4` + `grad_accum=4`,
     real-time so it can drive the local demo. Its SSL backbone ties to the repo's SimCLR theme,
     and its "small-dataset fine-tuning" design *is* the low-label story.
   - **Foil:** **LocateAnything-3B** (NVIDIA, ~May 2026) — a 3B vision-language open-vocab
     detector, used **zero-shot, Colab-side only**, to quantify the foundation-model X-ray
     out-of-distribution gap. Too heavy for the local demo.
   - RF-DETR is the engine the user *learns to drive*; the user's pseudo-labeling and temporal
     logic stay hand-written and transparent.
7. **Datasets:**
   - **Detection (boxes):** PIDray (~47k, boxes, openly downloadable, concealed-item subset) is
     the primary candidate; OPIXray (8,885 boxes) is an alternative but requires a signed access
     form emailed to the dataset authors, so it is slower.
   - **SSL pretraining pool:** SIXray (~1M images, easy access on Kaggle) as the large
     *unlabeled* pool. Boxes exist only on its ~8,929 positives.

## 3. Non-negotiable: keep the user in the loop (not a blackbox)

The user must understand the setup and code; this is a first-class deliverable.

- **Incremental build with review checkpoints** — small reviewable chunks; explain *what* and
  *why* at each checkpoint before moving on.
- **EXPLAINER per component** — plain-language docs tied to actual code lines (e.g. NT-Xent
  loss, how DETR predicts boxes, how the temporal aggregator gates decisions), under
  `docs/explainers/`.
- **Interactive notebooks** (matching the `archive/` style) reproducing key steps so the user
  can run and inspect them.
- **Guided code walkthrough** after each phase; the user drives the reproduction.

## 4. Architecture — phased crawl → walk → run

Each phase leaves a **working, understood artifact** before the next, riskier layer is added.
The low-label thesis spans **image-level (SimCLR) → box-level (pseudo-boxes)**.

### Phase 0 — image-level baseline (existing code, ported)
- Add a **ResNet18 encoder** option to `src/simclr_hpl/models.py` (torchvision already a dep);
  the current 3-layer `Encoder` is sized for 28px MNIST and will not suit X-ray. Keep the
  `EncoderClassifier` / probe interfaces so `training.py` and the `transfer-benchmark` CLI work
  unchanged.
- New X-ray dataset loader in `src/simclr_hpl/data.py`, following the existing MVTec loader
  pattern.
- New `configs/xray_screening.yaml`.
- Run existing `simclr-train` / `transfer-benchmark` / pseudo-labeling on X-ray for cheap early
  results and the image-level low-label baseline.

### Phase 1 — detection with RF-DETR (full labels)
- Integrate RF-DETR via a thin wrapper module (e.g. `src/simclr_hpl/detection/`).
- Fine-tune Nano/Small on PIDray/OPIXray on Colab T4 (Roboflow's fine-tuning notebook as a
  starting point).
- Write the DETR explainer.

### Phase 2 — the contribution: label-efficient detection
- Few-label fine-tune + **pseudo-box** semi-supervised expansion: confident detections become
  pseudo-boxes, confidence-gated and iterative — the repo's pseudo-labeling idea at the box
  level.
- Add the **LocateAnything-3B zero-shot foil** (Colab-side) to quantify the X-ray OOD gap.
- Metrics: mAP, sample-efficiency curves (labels vs mAP), pseudo-box precision,
  **recall-at-fixed-precision** (missing a threat is the costly error), multiclass confusion
  matrix, calibration (ECE).

### Phase 3 — temporal demo
- **Shared temporal inference core:** a new `TemporalAggregator` that ingests per-frame
  detections for a tracked object across the ~5s pass, accumulates evidence, and emits the gated
  decision (CLEAR/FLAG/REVIEW). It mirrors the threshold logic in
  `business.compute_review_queue_metrics` and the pseudo-label confidence gate — the **same
  principle in training, offline eval, and deployment**. Simple IoU/centroid tracking associates
  boxes across frames. Reuses `config.load_config`, `utils.{resolve_device, load_checkpoint,
  save_json}`.
- **Gradio app** (`src/simclr_hpl/cli/screening_demo.py` + console script), deployable to HF
  Spaces for a shareable link. Two modes (synthesized X-ray pass-through, live webcam) both feed
  the `TemporalAggregator`. UI shows a live confidence trace over the pass, a decision banner, a
  running **auto-decision-rate vs review-rate** dashboard (the `business.py` metrics as product
  UI), and boxes drawn on detected items.

## 5. The narrative thread (portfolio through-line)

The **same confidence-gated decision principle runs end-to-end**:
- *Training:* accept a pseudo-label/pseudo-box only when confident.
- *Offline eval:* auto-decide vs route to a human (`compute_review_queue_metrics`).
- *Deployment/demo:* auto-decide a bag only when accumulated confidence over the pass is high,
  else send to human review.

One idea, three places — pseudo-labeling, review queue, and live screening.

## 6. Reused existing code (do not reinvent)

- `training.py`: `pretrain_simclr`, `train_classifier`, `evaluate_classifier`,
  `generate_pseudo_labels`, `NTXentLoss`.
- `models.py`: `EncoderClassifier`, `ProjectionHead`, probe heads (extend with ResNet18, keep
  interfaces).
- `business.py`: `compute_review_queue_metrics` (the decision gate the demo mirrors).
- `data.py`: MVTec loader pattern for the X-ray loaders.
- `cli/transfer_benchmark.py`: random-vs-SimCLR benchmark, reused for Phase 0.
- `config.py`, `utils.py`, `visualization.py`, `plot-results` CLI (extend for
  mAP/recall@precision/ECE figures).

## 7. Honesty notes (state openly in the write-up)

- RGB webcam ≠ X-ray sensor — production swaps the sensor; the pipeline is identical.
- Synthesized pass-through *simulates* line-scan from real stills; it is not a real moving X-ray.
- LocateAnything-3B is used zero-shot as a baseline foil, expected to be weak on X-ray (OOD) —
  that is the point of including it.
- Portfolio-grade accuracy, not a certified detector.

## 8. Decomposition (each gets its own implementation plan)

1. **Sub-project 1 (build first):** Phase 0 image-level X-ray baseline.
2. **Sub-project 2:** Phases 1–2 RF-DETR detection + label-efficient/pseudo-box + LocateAnything
   foil + detection metrics.
3. **Sub-project 3:** Phase 3 temporal core + Gradio demo + HF Spaces deploy.

Each sub-project ships its EXPLAINER + notebook + walkthrough (the learning track).

## 9. Verification

- **Phase 0:** `uv run pytest` green (add a ResNet18 shape test); `uv run simclr-train --config
  configs/xray_screening.yaml` runs; `transfer-benchmark` produces sample-efficiency figures via
  `plot-results`; `uv run ruff check .` clean; CI unchanged and passing.
- **Phases 1–2:** RF-DETR fine-tune reproduces on Colab T4; mAP + recall@precision +
  pseudo-box-precision + confusion-matrix + ECE figures generated; LocateAnything-3B zero-shot
  numbers logged as the baseline.
- **Phase 3:** launch the Gradio app locally on CPU with the downloaded RF-DETR Nano checkpoint;
  run a synthesized X-ray pass and a webcam pass end-to-end; confirm the confidence trace updates
  over ~5s, boxes render, and the gated decision + auto-rate dashboard work; optionally deploy to
  HF Spaces and confirm the public link.
- **Learning track:** each phase has an EXPLAINER doc + runnable notebook; the user completes a
  walkthrough.

## References

- RF-DETR — Roboflow, ICLR 2026: https://github.com/roboflow/rf-detr
- LocateAnything-3B — NVIDIA: https://huggingface.co/nvidia/LocateAnything-3B
- SimCLR — https://arxiv.org/abs/2002.05709
- Semi-/weakly-supervised X-ray detection: I²OL-Net (arXiv 2412.03811), BCR-Net (arXiv
  2412.18918), Mix-Paste (arXiv 2501.01733)
- Datasets: OPIXray / HiXray / PIDray / SIXray (see XrayVision benchmark,
  https://github.com/NeelBhowmik/xrayvision-benchmark)
