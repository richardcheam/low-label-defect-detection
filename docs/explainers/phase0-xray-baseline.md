# Phase 0 explainer: the SimCLR + probe baseline on X-ray images

This explainer walks through the Phase 0 pipeline (`src/simclr_hpl/cli/xray_screening.py`)
function by function, so that running `uv run xray-screening --config configs/xray_screening.yaml`
isn't a black box. It assumes you're comfortable with Python and PyTorch basics, but newer
to self-supervised learning (SSL).

The big idea of Phase 0: **learn useful image features from a large pool of *unlabeled*
X-ray images (SimCLR), then check whether those features are good enough that a tiny,
frozen classifier ("probe") trained on a *small* labeled subset can already tell
threat from no-threat.** If the probe does well, that's evidence the representations
are doing the heavy lifting — which is exactly what you want when labels are expensive.

## 1. Why ResNet18 replaces the small `Encoder` for X-ray

The repo's original `Encoder` class (`src/simclr_hpl/models.py:8-41`) is a small
hand-rolled CNN built for MNIST: three conv blocks (`layer1`/`layer2`/`layer3`,
each `Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout`), an `AdaptiveAvgPool2d((4, 4))`,
and a flatten — landing on `output_dim = 4096` (`models.py:9`). It defaults to
`input_channels=1` because MNIST digits are 28x28 grayscale.

X-ray baggage scans are a different beast: larger, RGB (3-channel, even if the "color"
is a pseudo-color rendering of scan density), and structurally more complex than
handwritten digits. Re-using the small `Encoder` would mean either down-scaling images
to 28x28 (throwing away detail) or hacking the conv stack — neither is a good fit.

Instead, Phase 0 adds `ResNet18Encoder` (`models.py:44-58`), which wraps
`torchvision.models.resnet18`. Two details matter:

- **`backbone.fc = nn.Identity()`** (`models.py:53`). A normal ResNet18 ends in a
  `Linear(512, 1000)` layer for ImageNet's 1000 classes. Replacing that final layer
  with `nn.Identity()` (a no-op passthrough) turns the network from "an ImageNet
  classifier" into "a 512-dimensional feature extractor" — `forward` (`models.py:57-58`)
  now returns the pooled 512-d feature vector instead of class logits. That's exactly
  the shape SimCLR and the probes expect.
- **`output_dim = 512`** (`models.py:55`) replaces the small `Encoder`'s
  `output_dim = 4096`. Both encoders expose `.output_dim` as a class/instance attribute,
  which is the contract that lets `ProjectionHead`, `LinearProbe`, and `MLPProbe` stay
  encoder-agnostic — they just read `encoder.output_dim` to size their first linear layer.

The `build_encoder` factory (`models.py:61-71`) picks between the two by name:
`build_encoder("small", input_channels=1)` gives you the MNIST `Encoder`;
`build_encoder("resnet18", input_channels=3)` gives you `ResNet18Encoder`. The
`xray_screening` CLI calls it with whatever `model.encoder` / `model.input_channels`
say in the YAML config (`cli/xray_screening.py:89-93`; see `configs/xray_screening.yaml`,
which sets `encoder: resnet18` and `input_channels: 3`). Note `ResNet18Encoder` even
supports non-RGB input by swapping `conv1` for a differently-shaped `Conv2d`
(`models.py:49-52`), though Phase 0 doesn't exercise that path.

## 2. How SimCLR learns without labels

SimCLR's idea: **make two randomly-augmented "views" of the same image attract each
other in feature space, while every other image in the batch repels them.** No labels
required — the "positive pair" signal comes entirely from "these two crops/jitters came
from the same source image."

Two views are produced by `ContrastiveViewDataset` (`data.py:31-41`): its `__getitem__`
ignores the label and applies the *same stochastic transform* twice to the same image,
returning `(transform(image), transform(image))` — because the transform is randomized
(crops, flips, color jitter, etc.), the two outputs differ even though they came from
one source. The transform itself is `build_rgb_simclr_transform` (`data.py:175-194`):
resize, `RandomResizedCrop`, horizontal/vertical flips, `ColorJitter`, `RandomGrayscale`,
then `ToTensor` + ImageNet normalization. These augmentations are deliberately aggressive —
SimCLR only works if the two views are different enough that the network has to learn
*semantic* similarity rather than literally memorizing pixels.

The matching loss is `NTXentLoss` ("normalized temperature-scaled cross-entropy",
`training.py:12-36`). Conceptually:

1. **Normalize** every embedding to unit length (`F.normalize(..., dim=1)`,
   `training.py:19`) so that a dot product between two vectors becomes their
   **cosine similarity** — a measure of *direction* (semantic alignment) rather than
   magnitude.
2. Stack the two batches of projected views (`z_i`, `z_j`, each size `batch_size`) into
   one `2 * batch_size` matrix and compute the full pairwise similarity matrix
   `representations @ representations.T` (`training.py:19-20`), then divide by the
   **temperature** (`training.py:21`). A low temperature (the default is `0.5`,
   set in `configs/xray_screening.yaml`'s `train.temperature`) sharpens the
   distribution — it makes the model care more about getting the *most* similar pair
   right relative to the rest, which in practice produces more separated, useful
   embeddings.
3. Build masks that pick out, for each of the `2 * batch_size` rows, exactly one
   **positive** entry (its paired view — the diagonal blocks set up in
   `training.py:24-30`) and treat everything else (excluding the self-similarity
   diagonal) as **negatives** (`training.py:33`).
4. Frame this as a classification problem: each row's "correct class" is its single
   positive among all the candidates (`logits = torch.cat([positives, negatives], dim=1)`,
   labels are all zeros because the positive is always placed first), and minimize
   `F.cross_entropy(logits, labels)` (`training.py:34-36`).

In short: **pull the positive pair's similarity up, push every other pair's similarity
down, all measured as cosine similarity scaled by temperature.**

`pretrain_simclr` (`training.py:39-73`) is the training loop that wires this together:
for each batch it runs both views through the (currently trainable) `encoder` and then
a small `projection_head` (`models.py:74-84`, a 2-layer MLP that maps `encoder.output_dim`
→ 64-ish dims for the contrastive loss — the projection head is a SimCLR detail; it's
discarded after pretraining and only the encoder is kept), computes `NTXentLoss`,
backpropagates, and steps the optimizer. It returns a history dict of per-epoch average
loss. In the CLI (`cli/xray_screening.py:103-112`) this trained encoder's weights are
then checkpointed to `xray_encoder.pt` — that's the artifact this whole phase exists to
produce.

## 3. Why we freeze the encoder and train only a probe

Once SimCLR pretraining finishes, the natural question is: *did the encoder actually
learn something useful?* The standard SSL evaluation answers this with a **linear probe**:
freeze the pretrained encoder entirely, bolt on a single linear layer, and see how well
*that tiny layer alone* can separate the classes using the frozen features.

`freeze_module` (`training.py:172-174`) does the freezing — it walks every parameter in
a module and sets `requires_grad = False`, so gradients never flow into (and never
update) the encoder during probe training. `LinearProbe` (`models.py:87-94`) then wraps
the frozen encoder with exactly one `nn.Linear(encoder.output_dim, num_classes)`. If this
single linear layer, with no hidden non-linearity, achieves strong accuracy, that's a
strong signal that the encoder already arranged the data into **linearly separable**
clusters — i.e., the representation itself (not a clever classifier head) is doing the
work. `MLPProbe` (`models.py:97-108`) is the slightly more forgiving sibling — it adds
one hidden `Linear → ReLU → Linear`, useful as a sanity check (and as a second number to
report) but the *linear* probe is the more meaningful "is the representation good"
diagnostic.

Crucially, this probe stage **is the evaluation, not the main training** — the real
"training" already happened during SimCLR pretraining on unlabeled data. The probe's job
is small and cheap on purpose: `trainable_parameters` (`training.py:177-178`) filters down
to just the unfrozen probe-head parameters, and `train_classifier`
(`training.py:130-169`) trains/validates that small head with ordinary supervised
cross-entropy for a handful of epochs. In the CLI (`cli/xray_screening.py:128-145`) this
loop runs twice — once for `LinearProbe`, once for `MLPProbe` — re-freezing the encoder
each time (`freeze_module(encoder)` at `cli/xray_screening.py:130`, since a fresh
optimizer/probe is built per iteration) and reporting both sets of metrics.

## 4. The low-label angle: two different sample budgets

This is the heart of the "low-label" thesis, and it shows up as **two separate
subsampling knobs feeding two separate stages**:

- `max_pretrain_images` (read at `cli/xray_screening.py:75-77`, via the small
  `_subsample` helper at `cli/xray_screening.py:46-49`) caps how many images go into
  the **unlabeled** SimCLR pretraining pool. This can — and should — be large, because
  SimCLR doesn't need labels at all; it just needs lots of raw images to learn good
  general-purpose visual structure from.
- `max_labeled_per_class`, consumed via `subsample_per_class`
  (`data.py:314-338`, called at `cli/xray_screening.py:114-116`), caps how many
  **labeled** examples *per class* are used to train the probe. `subsample_per_class`
  groups record indices by integer label, randomly samples down to `max_per_class`
  per group (deterministically, via a seeded `random.Random`), and returns them in
  stable original order — so the labeled budget stays small, balanced across classes,
  and reproducible.

The config (`configs/xray_screening.yaml`) sets these to `max_pretrain_images: 4000`
vs. `max_labeled_per_class: 500` — i.e., SimCLR sees up to 4000 unlabeled images, while
the probe is trained on at most 500 *labeled* examples per class (1000 total for the
binary problem). That asymmetry **is** the experiment: can a model that mostly learned
from unlabeled data reach good accuracy with only a small labeled budget? If yes, that's
the practical payoff of self-supervised pretraining — you spend your (expensive,
slow-to-collect) labeling budget sparingly, and let cheap unlabeled data carry most of
the representation-learning weight.

## 5. The binary threat/no-threat metric — and the through-line to deployment

`evaluate_classifier` (`training.py:76-98`) is what finally turns "frozen features +
small probe head" into a number: it runs the probe in `eval()` mode over a held-out test
loader, accumulates loss and the count of `logits.argmax(dim=1) == targets`, and returns
`{"loss": ..., "accuracy": ...}`. For Phase 0, "accuracy" answers a simple binary
question per image: *threat (label 1) or no-threat (label 0)?* — the labels come straight
from `load_binary_image_records`'s `class_to_label` mapping (`data.py:100-128`,
`configs/xray_screening.yaml`'s `data.class_to_label: {positive: 1, negative: 0}`).

Here's the through-line to where this project is headed: a single accuracy number is
useful for *judging representation quality*, but a deployed screening system needs to
do something more nuanced than "always trust the model's call." That's where
`compute_review_queue_metrics` in `src/simclr_hpl/business.py:4-52` comes in. It takes
the same kind of per-image outputs (`predictions`, `targets`, and per-prediction
**confidences** — see `collect_prediction_outputs`, `training.py:101-127`, which is the
sibling of `evaluate_classifier` that also records softmax confidence) plus an
`auto_decision_threshold`, and splits the stream into two buckets: predictions confident
enough to **auto-decide** (`confidence >= auto_decision_threshold`,
`business.py:23-25`) versus everything else, which is routed to the **human review
queue** (`business.py:27`). It then reports rates and accuracy/precision/recall split
across those two buckets (`auto_decision_rate`, `review_queue_rate`,
`auto_decision_accuracy`, `auto_defect_precision`, `overall_defect_recall`,
`overall_false_negative_rate`, ...).

The connection: Phase 0's probe accuracy tells you *whether the underlying model is any
good at all*. The confidence-gated, auto-decide-vs-review-queue logic in
`compute_review_queue_metrics` is the next layer up — it's the same "trust the model when
it's confident, defer to a human when it's not" principle that will drive the later
temporal-screening demo. Phase 0 builds (and measures) the model; this business-logic
layer is how that model's *confidence*, not just its raw accuracy, gets turned into an
operational decision rule.

## How to run it yourself

- **Synthetic smoke test (no data download, runs anywhere):**
  ```bash
  uv run pytest tests/test_xray_cli.py -v
  ```
  This builds a tiny synthetic red/blue image dataset on the fly, runs the *entire*
  pipeline (SimCLR pretrain → linear/MLP probe → metrics) on CPU, and asserts that
  `metrics.json` and `xray_encoder.pt` get written. It's the fastest way to confirm the
  whole wiring is correct without any real X-ray data.

- **Real run (needs the X-ray dataset under `data/sixray`, see `scripts/get_sixray.md`):**
  ```bash
  uv run xray-screening --config configs/xray_screening.yaml
  ```
  This is the actual Phase 0 experiment: SimCLR-pretrains a `ResNet18Encoder` on the
  unlabeled pool, evaluates linear and MLP probes on a small labeled subset, and writes
  `artifacts/xray_screening/{xray_encoder.pt,metrics.json}`.
