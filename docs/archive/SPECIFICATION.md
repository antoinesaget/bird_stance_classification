# Bird Annotation & Classification System — PhD-Grade Plan (Standalone)

## 0) Project goal (context)

We want a self-hosted system to annotate and model bird images.

Each image may contain zero or more birds. For each **usable** bird we want:
- a bounding box
- attributes:
  - activity: flying / foraging / standing
  - support surface: ground / water / air
  - legs (only meaningful if standing): one / two / unsure
  - back resting (only meaningful if standing): yes / no
  - readability: readable / unreadable

Additionally, each image must be labeled:
- image_status: has_usable_birds / no_usable_birds

We ultimately care about **standing + readable** birds, and mainly **legs one vs two** on **ground/water**. Unreadable birds are discarded from training and evaluation for most heads (see training rules).

Constraints:
- Must support both **RTX 3090 (Linux)** and **MacBook Pro M4 (macOS)** workflows
- Up to 3 annotators
- Self-hosted, open-source
- **Label Studio Community**
- **No MinIO/S3**
- **No SAM refinement**
- Detector (YOLO) used for pre-annotations (no fine-tuning for now)
- Active learning is **manual**, driven by **Model B** confidence/uncertainty (not YOLO)

---

## 1) System overview

Components:
1) Local raw image store (disk)
2) Label Studio Community (annotation UI)
3) Single ML backend service hooked into Label Studio (provides:
   - YOLO11m bbox pre-annotations
   - Model B attribute predictions for each bird crop
   - Model C image_status prediction)
4) Export normalization pipeline → Parquet
5) DuckDB-based dataset assembly
6) Training pipelines:
   - Model B: attribute classifier (multi-head)
   - Model C: image_status classifier (binary)
7) Manual active learning workflow based on Model B uncertainty (and optionally Model C later)

Optional later:
- FiftyOne (dataset QA)
- Weights & Biases (experiment tracking)

---

## 2) Hardware support strategy (RTX 3090 + MacBook Pro M4)

### RTX 3090 machine (primary training + inference)
- Runs Label Studio + ML backend + training jobs
- Fast iteration, heavy training, batched inference

### MacBook Pro M4 (secondary dev + light inference)
- Used for:
  - development
  - running Label Studio client remotely
  - small-scale experiments
  - optional local inference using Apple ML stack if desired
- Full training for Model B/C is expected to be more efficient on the 3090; Mac is “supported” for compatibility and smaller tests.

Implementation requirement:
- All code paths must run on both platforms (e.g., PyTorch/MPS on Mac, CUDA on 3090), but primary performance target is the 3090.

---

## 3) Storage layout (no MinIO, versioned + reproducible)

Full folder structure:

```text
/data/birds_project/
│
├── raw_images/                         # immutable source images (never modified)
│   ├── siteA/
│   │   ├── img_0001.jpg
│   │   └── …
│   ├── siteB/
│   └── …
│
├── metadata/
│   └── images.parquet                  # one row per image:
│                                       # image_id, filepath, datetime (optional), site_id, etc.
│
├── labelstudio/
│   ├── exports/                        # raw exports from Label Studio (immutable)
│   │   ├── ann_v001.json
│   │   ├── ann_v002.json
│   │   └── …
│   │
│   └── normalized/                     # deterministic normalized outputs from exports
│       ├── ann_v001/
│       │   ├── birds.parquet
│       │   └── images_labels.parquet
│       ├── ann_v002/
│       │   ├── birds.parquet
│       │   └── images_labels.parquet
│       └── …
│
├── derived/
│   ├── crops/                          # cached bird crops extracted from bboxes
│   │   ├── ann_v001/
│   │   │   ├── bird_000001.jpg
│   │   │   └── …
│   │   ├── ann_v002/
│   │   └── …
│   │
│   └── datasets/                       # training-ready dataset versions
│       ├── ds_v001/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   ├── test.parquet
│       │   └── manifest.json           # describes filters, annotation version, split rules
│       ├── ds_v002/
│       └── …
│
├── models/
│   ├── detector/                       # reference only (pretrained, not trained here)
│   │   └── yolo11m_pretrained/
│   │       └── weights.pt
│   │
│   ├── attributes/                     # Model B checkpoints
│   │   ├── convnextv2s_v001/
│   │   │   ├── config.yaml
│   │   │   ├── checkpoint.pt
│   │   │   └── metrics.json
│   │   ├── convnextv2s_v002/
│   │   └── …
│   │
│   └── image_status/                   # Model C checkpoints
│       ├── status_v001/
│       │   ├── config.yaml
│       │   ├── checkpoint.pt
│       │   └── metrics.json
│       ├── status_v002/
│       └── …
│
└── scripts/                            # reproducible pipeline scripts
├── export_normalize.py             # Label Studio JSON → normalized Parquet
├── make_crops.py                   # bbox → image crops
├── build_dataset_duckdb.sql        # dataset assembly queries
├── train_attributes.py             # Model B training
├── train_image_status.py           # Model C training
├── infer_batch.py                  # batch inference for active learning
└── utils/
```

 
### Storage rules

1. `raw_images/` is immutable  
   - never rename or overwrite files  
   - treat as permanent source of truth  

2. Label Studio exports are immutable  
   - never modify files inside `labelstudio/exports/`  
   - each export creates a new `ann_vXXX`  

3. All derived data is versioned  
   - normalized annotations: `ann_vXXX`
   - datasets: `ds_vXXX`
   - models: `*_vXXX`

4. Never overwrite previous versions  
   Always create new version directories.

5. Entire system must be reproducible from:
   - `raw_images/`
   - `metadata/images.parquet`
   - Label Studio export `ann_vXXX.json`
   - code in `/scripts/`

---

## 4) Annotation schema (Label Studio UX)

### 4.1 Image-level (required, no “uncertain”)
`image_status` (single choice):
- has_usable_birds
- no_usable_birds

Rationale:
- explicit negatives (no usable birds)
- avoids “empty annotation ambiguity” (missed labels vs true empty)

### 4.2 Object-level (per bird bbox)
Bounding box label: `Bird`

Attributes:

1) `readability` (required)
- readable
- unreadable

2) If readability == readable, also require:
- `activity`: flying / foraging / standing
- `support`: ground / water / air
- If activity == standing:
  - `resting_back`: yes / no
  - `legs`: one / two / unsure

Important:
- No “unsure” for activity/support/resting_back
- Only “legs” keeps an “unsure” option (even if readable, legs may still be ambiguous)

Discard policy:
- If a bird is labeled unreadable → it will be discarded from training/evaluation for activity/support/resting_back/legs (except readability head).
- Practically, unreadable birds exist mainly to prevent noise and to train the readability head.

---

## 5) Label Studio UX design (fast, consistent for ≤3 annotators)

Defaults:
- For a Bird region:
  - readability default = readable
  - resting_back default = no (only if standing)
- Conditional UI:
  - activity/support shown only if readability=readable
  - resting_back/legs shown only if readability=readable AND activity=standing

Annotator workflow per image:
1) Set image_status
2) Review auto bboxes (YOLO pre-annotations)
3) Adjust bboxes if needed
4) For each bird:
   - mark readability
   - if readable: fill activity + support
   - if readable & standing: fill legs + resting_back

Emphasis:
- speed + consistency
- avoid forcing labels that won’t be used

---

## 6) ML integration into Label Studio (single hooked backend)

Use one ML backend service integrated with Label Studio (not multiple separate apps from the annotator perspective).

The backend provides:
- YOLO11m detections as bbox pre-annotations (no fine-tuning)
- Model B attribute predictions for each proposed bbox crop
- Model C image_status prediction

Behavior when opening a task:
1) If no existing predictions:
   - run YOLO11m → propose bboxes
   - for each bbox: run Model B on crop → propose readability/activity/support/(standing→legs/resting_back)
   - run Model C on whole image → propose image_status
2) Return everything as predictions with confidence scores (for annotator assistance)

Note:
- No SAM refinement endpoints for now.

---

## 7) Normalization pipeline → Parquet (deterministic, versioned)

For each Label Studio export `ann_vXXX.json`, produce:

### images_labels.parquet (one row per image)
Columns:
- annotation_version
- image_id
- image_status

### birds.parquet (one row per bird)
Columns:
- annotation_version
- image_id
- bird_id (e.g., `${image_id}:${index}`)
- bbox (x,y,w,h) normalized + optional pixel coords
- readability
- activity (nullable if unreadable)
- support (nullable if unreadable)
- legs (nullable unless readable & standing)
- resting_back (nullable unless readable & standing)

This normalized representation is the base for all datasets/training.

---

## 8) Dataset assembly with DuckDB

Use DuckDB to build dataset versions `ds_vXXX` from:
- `metadata/images.parquet`
- `labelstudio/normalized/ann_vXXX/*.parquet`

Output:
- `train.parquet`, `val.parquet`, `test.parquet`
- `manifest.json` describing:
  - source annotation version
  - filtering rules
  - split policy
  - counts per label

Split policy (recommended):
- stratify by site/date if applicable to avoid leakage
- otherwise a stable hash split by image_id

---

## 9) Models

### Model A (pretrained detector, no fine-tuning now)
- YOLO11m pretrained
- used only for bbox proposals to speed annotation

### Model B (trained): Bird attribute classifier (multi-head)
Backbone:
- ConvNeXtV2-Small pretrained (default)

Inputs:
- crop from bbox with margin (~1.2×)
- resize to 384–448px

Heads (outputs):
1) readability (binary): readable/unreadable
2) activity (3-way): flying/foraging/standing
3) support (3-way): ground/water/air
4) resting_back (2-way): yes/no
5) legs (3-way): one/two/unsure

Training/evaluation masking rules (critical):
- readability head: trained on all bird crops
- activity head: trained/eval only if GT readability == readable
- support head: trained/eval only if GT readability == readable
- resting_back head: trained/eval only if GT readability == readable AND GT activity == standing
- legs head: trained/eval only if GT readability == readable AND GT activity == standing

Rationale:
- do not penalize predictions for labels that are irrelevant or undefined

Training schedule:
- Phase 1: freeze backbone, train heads
- Phase 2: unfreeze last stage(s), fine-tune lightly

Metrics:
- readability: accuracy/F1
- activity/support: accuracy/F1 on readable subset only
- resting_back/legs: accuracy/F1 on readable & standing subset only
- legs confusion matrix (one vs two vs unsure)

### Model C (trained): image_status classifier (binary)
Goal:
- predict `has_usable_birds` vs `no_usable_birds`

Input:
- whole image (or downscaled)
- optionally use a lightweight backbone (ConvNeXt-Tiny / EfficientNet-like) for speed

Training data:
- image_status labels from Label Studio exports
- should be robust to “false positives” by YOLO (since YOLO is not the authority for image_status)

Usage:
- prefill image_status in Label Studio to speed annotation
- later can support batch triage (find likely “no usable birds”)

---

## 10) Manual active learning (driven by Model B confidence)

Active learning is manual and based on **Model B uncertainty**, not on YOLO.

Batch creation workflow:
1) Run inference on a pool of unlabeled images:
   - YOLO proposes bboxes
   - Model B predicts readability/activity/support/standing→legs/resting_back per bbox
2) Select images for annotation using heuristics based on Model B:
   - high uncertainty on readability (near 0.5)
   - high probability of standing & readable but low confidence on legs
   - disagreement patterns (e.g., standing high but legs uncertain)
   - diversity: include a small random slice each cycle to prevent bias
3) Create Label Studio batches/projects from selected images

Annotation cycles:
- start with a random bootstrap batch
- then move to uncertainty-driven batches once Model B is reasonable

---

## 11) Optional add-ons (not required now)

### FiftyOne (optional later)
Add for dataset QA and error analysis:
- filter/slice: standing+readable but legs errors
- find duplicates and clusters
- visualize predictions vs GT

### Weights & Biases (optional later)
Add for experiment tracking:
- log configs, metrics, checkpoints
- log dataset manifests
- keep model/dataset lineage tidy

---

## 12) Reproducibility rules

- Raw images immutable
- Never overwrite exports/normalized/datasets/models
- All datasets produced by deterministic DuckDB queries + manifests
- All training runs produce:
  - config
  - metrics
  - checkpoint
  - dataset version reference

Everything must be rebuildable from:
- raw_images/
- images.parquet (metadata)
- ann_vXXX.json exports
- code + dataset manifests
