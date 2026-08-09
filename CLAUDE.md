# Tomato Leaf Disease Detector — Build Plan (as built)

## Status: complete

The project is built, trained and verified. Measured results, all from one real Colab run
on a T4 and reproduced locally:

| | |
|---|---|
| Test macro F1 | **0.9495** |
| Test accuracy | 0.9488 (3,343 images) |
| Stage 1 (frozen backbone) | 0.8341 best val accuracy |
| Stage 2 (top 35% unfrozen) | 0.9422 best val accuracy |

See `README.md` for the full write-up. This file records the plan and the decisions that
changed during the build.

## Decisions that changed from the original plan

- **Fine-tuning went from "out of scope" to the central result.** The plan specified a frozen
  backbone only. Stage 1 plateaued at 0.8341 with training accuracy *below* validation and
  both losses level — underfitting, not overfitting, meaning the frozen ImageNet features
  were the ceiling. Unfreezing the top 35% at `lr=1e-5` added **+10.81 points**. The frozen
  run is kept in the README as a documented ablation rather than deleted.
- **Corrupt-file handling had to grow a second branch.** The plan (and the original
  `is_decodable`) assumed one truncated PNG. The archive also ships **two WebP files with
  `.jpg` extensions**, which Pillow opens happily but `tf.io.decode_image` rejects — they
  crashed local evaluation with `Unknown image file format`. These are now *re-encoded in
  place* rather than deleted, so split counts stay identical to the trained run. The
  notebook's `%%writefile` mirror of `prepare_data.py` carries the same fix.
- **The low-confidence guard does not do what the plan assumed.** The plan treated a 50%
  softmax floor as a safeguard against non-leaf input. Measured: a solid grey square returns
  `Late_blight` at 98.2%, a drawn face returns `Tomato_mosaic_virus` at 99.2%, against a
  real-leaf median of 0.998. The distributions overlap completely and entropy is no better,
  so no threshold separates them. The flag is retained for ambiguous *leaf* photos, and the
  UI and README now state plainly that it does not detect wrong subjects.
- **The app grew from one view to four tabs**, adding Model Performance (metrics read from
  `reports/` at runtime, never hardcoded), Explainability (Grad-CAM, conv filters, feature
  maps), and Method & Limitations. A **light/dark toggle** was added later: both themes come
  from one `PALETTES` dict of semantic tokens that drives the CSS *and* the matplotlib
  figures. The theme is read from `session_state` at the top of the script, above the
  stylesheet, while the toggle widget itself is drawn far below in the sidebar — legal
  because a flip reruns the script from line 1. Chart functions take `theme` as an explicit
  argument so it forms part of the `cache_data` key; without it, flipping the toggle would
  serve the other theme's cached PNG. `reports/training_curves.png` is pre-rendered on a
  white canvas and cannot follow the palette, so it sits on an explicit white card.
- **Local training was added to the loop.** `data/` is now built locally so the app has real
  test images and evaluation can be re-run; the local split hashes identically to Colab's.
- **The app needs an interpreter guard; bare `streamlit run` used to fail on this machine.**
  It died with `ModuleNotFoundError: No module named 'tensorflow'` *even with `(tf-uetm)` in
  the prompt*. `~/.bashrc:123-125` prepends pyenv's shims to `PATH` after conda's block, so
  `python`, `streamlit` **and `conda run -n tf-uetm`** all resolve to pyenv's Python 3.14,
  which has Streamlit but no TensorFlow — and never will, since TF ships no 3.14 wheels.
  `CONDA_PREFIX` and the prompt are correct; only `PATH` is wrong, which is why the failure
  reads as a broken env when the env is fine. `src/interpreter_guard.py` now runs before any
  TF-touching import, finds an interpreter that has TensorFlow, and `execve`s the same
  command under it — the server restarts itself on the same port. Two non-obvious details:
  the guard must read `/proc/self/cmdline`, because Streamlit rewrites `sys.argv` to the
  *script's* args and the server flags are invisible there; and it must close inherited
  socket FDs before `execve`, or the replacement process finds the port held by its own
  inherited descriptor and exits with `Port 8501 is not available` (this was observed, not
  theorised). `run_app.sh` remains as the explicit no-restart path.

- **Streamlit Community Cloud needed three changes, one of which is not in the repo at all.**
  The first deploy failed during dependency resolution — the container ran **Python 3.14.7**
  and TensorFlow publishes wheels for 3.10–3.13 only, so `uv` and `pip` both reported no
  matching ABI tag. Nothing in `requirements.txt` can fix that. The Python version is chosen
  in the deploy-time **Advanced settings** dropdown and **cannot be changed in place** — the
  app must be deleted and redeployed — and `runtime.txt` is a Heroku convention that
  Community Cloud ignores, so pinning it from the repo is impossible. That step stays the
  human's. What *is* in the repo: `requirements.txt` now installs **`tensorflow-cpu`**
  (252 MB) instead of `tensorflow` (645 MB, drags in the CUDA stack) via a `sys_platform`
  marker, because `tensorflow-cpu` ships Linux and Windows wheels **only** and a bare swap
  would break macOS clones. And `interpreter_guard.py` gained a `_managed_host()` check:
  its interpreter scan is a local-machine fix, so on a hosted runner it now skips the scan
  and raises the actual cause instead of "searched: (none found)".
- **Streamlit's own top bar was covering the page title.** `[data-testid="stHeader"]` is
  fixed-position and paints itself from `config.toml`, which can declare only one base — so
  under the dark palette it rendered as a light band across the top ~3.5rem of the page. It
  is now forced transparent, with `.block-container` padding raised 2.2rem → 4.2rem so the
  title clears the toolbar buttons that still live there. The pre-existing `#MainMenu,
  footer` rule was dead code: current Streamlit uses `data-testid` nodes, verified by
  grepping the installed bundle (`stHeader` 17 hits, `stAppDeployButton` 2,
  `stAppViewBlockContainer` 0).
- **The sidebar carries the leaf mark from the browser tab.** Emoji in a palette-driven
  badge, not an SVG or a remote image — no asset file, no network fetch, so it renders the
  same offline and on the deployed container.

## Verification performed

1. `prepare_data.py` → per-class table, 32,534 images, counts sum correctly, no filename in
   two splits. ✅
2. `get_datasets()` → 11 classes, correct shapes. ✅
3. `build_model(11)` → output `(None, 11)`; MobileNetV2 frozen in stage 1; at stage 2,
   154-layer backbone, cut at layer 100, all 52 BatchNorm layers frozen, moving statistics
   verified unmoved, trainable params 14,091 → 1,853,707. ✅
4. Notebook ran end to end on a T4 → `artifacts.zip`. ✅
5. `evaluate_model.py` locally → macro F1 0.9495 / accuracy 0.9488, **identical** to Colab in
   every per-class figure; both `split_manifest.json` files hash to the same SHA-256. No
   missing-advice warnings; all 11 classes resolve with real EN + UR text. ✅
6. App verified through Streamlit's `AppTest` harness: 0 exceptions, 4 tabs
   (`Diagnose`, `Model Performance`, `Explainability`, `Method & Limitations`), sample-load,
   matrix toggle and Grad-CAM counterfactual paths all clean. Correctly identifies a held-out
   `Late_blight` image at 100%. Metrics render from `reports/` (0.9495 / 0.9488 both present),
   and Urdu renders as real script with RTL — 164 Arabic-script runs, `dir="rtl"` applied. ✅
   `run_app.sh` re-verified from `/tmp` with pyenv shims *first* on `PATH` — the exact
   condition that broke the bare launch — and it still resolved the conda interpreter and
   repo-root cwd. Both its guards exit 1 with instructions (missing interpreter, interpreter
   without TF). ✅
   `interpreter_guard.py` verified against the real failure: launched with **pyenv's**
   `streamlit` binary, then driven by a websocket client (`BackMsg.rerun_script`) because
   Streamlit only executes the script when a client connects — an HTTP GET returns the page
   shell without ever running it. The server restarted itself under
   `envs/tf-uetm/bin/python3.11` on the same port and rendered cleanly; a second run through
   the restarted server produced no errors. The first attempt failed with `Port 8611 is not
   available`, which is what exposed the inherited-socket problem. ✅
7. Negative check with non-leaf inputs → **failed as designed, documented as a limitation**
   (see above). This is the one verification step whose result contradicted the plan.

---

## Original plan, for reference

## Context

`plan.md` describes a 2-day MVP for an Applied AI & ML midterm: a transfer-learning
tomato leaf disease classifier (Keras `Sequential` + frozen MobileNetV2) served through a
custom-styled Streamlit app. Nothing exists in the repo yet except `plan.md`.

The original plan is sound in shape — right stack, right scope, correct layer ordering,
honest about what's manual. This revision keeps all of that and fixes eight concrete
problems that would each have cost real debugging time or silently produced misleading
results.

## Decisions confirmed

- **Split:** preserve the archive's own `train`/`valid` boundary (no random re-pooling).
- **Training:** Google Colab, T4 GPU. The notebook must be **fully self-contained** —
  unzip, prepare, train, evaluate, save. Claude builds it; the human runs it and sends
  back the trained model.
- **Fine-tuning:** out of scope. Frozen backbone only, as originally planned.

---

## What changed from `plan.md`, and why

### 1. The dataset is already split — don't assume flat class folders
`plan.md` assumes a raw `raw_data/<class>/*.jpg` download. The Kaggle "Tomato Disease
Multiple Sources" archive ships **pre-split as `train/` and `valid/`**, each holding the
class subfolders. `prepare_data.py` must detect which layout it got rather than assume.

**Fix:** Step 0 inspects `archive.zip` before any pipeline code is written.
`prepare_data.py` then branches on a small `detect_layout()` check.

### 2. Random re-splitting leaks near-duplicates and produces a fake accuracy number
This dataset aggregates *multiple sources* (lab PlantVillage images plus field photos) and
contains near-duplicate images. Pooling `train/` + `valid/` and re-shuffling 70/15/15 puts
near-duplicates on **both sides of the split**, so test accuracy measures memorisation.
You get ~99% that means nothing — bad in a report where limitations are graded.

**Fix (your choice, now the default):** archive `train/` → `data/train`; archive `valid/`
split in half, stratified per class, into `data/val` and `data/test`. Test images were
never in the training pool. Effective ratio lands near 80/10/10 rather than 70/15/15,
which is the honest trade and is stated in the README.

### 3. The split must be byte-identical in Colab and locally, or local eval is invalid
New risk created by training remotely: the notebook splits `valid/` in Colab, and you'd
also split locally to get demo images for Streamlit. If those two splits disagree, images
the model trained on end up in your local `data/test/` and local evaluation silently
overstates performance.

**Fix:** the split is made deterministic — filenames sorted before splitting, fixed
`seed=42`, no reliance on filesystem iteration order — so both machines produce the same
assignment. `prepare_data.py` also writes `data/split_manifest.json`, and the notebook
saves its copy alongside the checkpoint so the two can be diffed to confirm. Additionally
**evaluation runs in Colab** and its `reports/` come back with the model, making the
authoritative numbers independent of any local split.

### 4. `src/.streamlit/config.toml` will be silently ignored
Streamlit resolves `./.streamlit/config.toml` against the **current working directory**,
not the script's location. `streamlit run src/streamlit_app.py` from the repo root never
reads `src/.streamlit/config.toml` — the theme just doesn't apply, with no error.

**Fix:** config lives at repo-root `.streamlit/config.toml`. All internal paths resolve
from `ROOT = Path(__file__).resolve().parents[1]` so code is CWD-independent, and the
README documents exactly one launch command.

### 5. The manual `class_names` copy-back step is unnecessary and fragile
`plan.md` asks you to read `class_names` out of the Colab output and hand-edit
`advice.py`'s keys, and correctly notes a mismatch "means predictions silently show no
advice text, not an error." That's a silent-failure mode designed in on purpose — and it
gets worse when the model is trained on a different machine.

**Fix:** the notebook writes `models/class_names.json` next to the checkpoint; `infer.py`
reads it. Class order from `image_dataset_from_directory` is alphabetical and
deterministic, so this is a reliable contract. `advice.py` exposes `get_advice(class_name)`
with a **normalised** lookup (lowercase, strip non-alphanumerics) plus an explicit
fallback. A naming mismatch now degrades to a visible "advice not available for this class
yet" instead of a blank card, and `evaluate_model.py` warns at startup listing any class
with no advice entry. This deletes manual step #4 from your list entirely.

### 6. The class list in `advice.py` is missing a class
`plan.md`'s placeholder list has **10** entries against a dataset of **11 classes**. The
commonly-missed one is `powdery_mildew`, present here but absent from the classic
PlantVillage tomato set the placeholder names came from.

**Fix:** write advice for all 11, with exact names taken from the extracted archive in
Step 0 — the archive is ground truth, not this plan and not memory.

### 7. Class imbalance is unaddressed, and accuracy alone hides it
Class counts here are uneven. Training on raw counts with plain `accuracy` as the headline
metric lets minority classes fail invisibly.

**Fix:** compute `class_weight` via `sklearn.utils.class_weight.compute_class_weight` and
pass it to `model.fit()`. Evaluation leads with **macro F1** alongside accuracy, plus a
per-class table — a better result for the report for two lines of code.

### 8. Missing scaffolding
`plan.md` refers to gitignored directories but never creates `.gitignore`; this isn't a git
repo at all; eval artifacts go "somewhere the report can reference" with no location; and
the tree says `crop-disease-mvp/` while the real directory is `tomato-disease-detector/`.

**Fix:** add `.gitignore`, a `reports/` directory, and use the real name throughout. Copy
the final plan to `CLAUDE.md` as `plan.md` itself suggests.

### 9. `.cache()` on the training set will OOM the Colab session
A standard `image_dataset_from_directory(...).cache().prefetch(AUTOTUNE)` — the idiom most
tutorials use, and what I'd have written from `plan.md`'s "don't hand-roll" instruction —
caches decoded **float32** tensors in RAM. For this dataset that is roughly
20,000 × 224 × 224 × 3 × 4 bytes ≈ **12 GB**, against a free Colab T4 instance's ~12.7 GB.
The session dies partway through epoch 1 with an unhelpful "your runtime has crashed",
after you've already spent time uploading 1.4 GB.

**Fix:** no in-memory `.cache()` on train — `prefetch(AUTOTUNE)` only (disk cache via
`.cache(filename=...)` is the alternative if epoch time proves I/O-bound). Val and test are
~10× smaller and cache safely. Called out in `dataset.py` with a comment so it isn't
"helpfully" re-added later.

### Smaller robustness fixes folded in
- `infer.py`: `ImageOps.exif_transpose()` + `.convert("RGB")` before resize — phone uploads
  are routinely EXIF-rotated, RGBA, or greyscale, all of which break a bare `np.array(img)`.
- `streamlit_app.py`: `@st.cache_resource` on model load, or it reloads on every interaction.
- Augmentation runs **before** `Rescaling`, on 0–255 data. `RandomContrast` on already
  normalised [-1, 1] input behaves incorrectly. `plan.md` has this right; keeping it
  explicit so it isn't reversed.
- `EarlyStopping(patience=3, restore_best_weights=True)` alongside `ModelCheckpoint`.
- Low-confidence guard in the UI: softmax is always confident about *something*, including
  a photo with no leaf in it. Below threshold, show a caveat rather than a diagnosis.

### Deliberately kept from `plan.md`
Keras `Sequential` throughout, MobileNetV2 frozen backbone, preprocessing baked into the
saved model, the "field diagnostic scan" UI direction and palette, bilingual EN/UR advice,
2-day scope, honest-limitations framing.

---

## Repo structure to build

```
tomato-disease-detector/
├── CLAUDE.md                   # copy of this plan
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml             # repo root — NOT src/ (see fix #4)
├── data/                       # gitignored — created by prepare_data.py
├── models/                     # gitignored — checkpoint + class_names.json
├── reports/                    # gitignored — evaluation artifacts from Colab
├── notebooks/
│   └── train.ipynb             # self-contained Colab notebook
└── src/
    ├── prepare_data.py
    ├── dataset.py
    ├── model.py
    ├── advice.py
    ├── infer.py
    ├── evaluate_model.py
    └── streamlit_app.py
```

No `src/train.py` — training lives in the notebook, per your decision.

---

## Build sequence

### Step 0 — Inspect the archive (blocking, do first)
The sandbox classifier was down for this entire planning session, so **nothing below has
been verified against your actual files** — no directory listing, no archive inspection, no
GPU probe. Everything about the dataset here comes from `plan.md` and background knowledge
of this Kaggle dataset. First build action is therefore:

- Unzip and read the top-level layout (`train/`+`valid/`? flat class dirs? extra nesting?)
- Record exact class directory names, verbatim, and the count (expect 11)
- Per-class image counts, to size the imbalance and compute class weights
- Total count and extracted size

Everything downstream — split code, `advice.py` keys, class weights — keys off this.
**Do not write `advice.py` keys from memory.** If Bash is still unavailable at build time,
`!unzip -l archive.zip | head -50` typed directly in the session answers the layout and
class-name questions without needing my tools.

### Step 1 — Scaffolding
`requirements.txt` (`tensorflow`, `streamlit`, `pillow`, `scikit-learn`, `matplotlib`,
`numpy` — no torch), `.gitignore` (`data/`, `models/`, `reports/`, `archive.zip`,
`__pycache__/`, `.ipynb_checkpoints/`), `.streamlit/config.toml` with the sage/forest/ochre
base theme.

### Step 2 — `src/prepare_data.py`
CLI: `--src`, `--out data`, `--seed 42`. Detects archive layout, then builds
`data/train|val|test/<class>/` using the preserve strategy: archive `train/` → `data/train`;
archive `valid/` → stratified 50/50 into `data/val` + `data/test`.

**Determinism is load-bearing** (fix #3): sort filenames before splitting, seed the RNG, never
depend on filesystem order — Colab and your laptop must produce identical assignments.
Writes `data/split_manifest.json` (per-class counts + the test filename list) so the two
can be diffed. Prints a per-class count table. Skips if output exists unless `--force`.

### Step 3 — `src/dataset.py`
`get_datasets(data_dir="data", image_size=(224,224), batch_size=32)` →
`(train_ds, val_ds, test_ds, class_names)` via `tf.keras.utils.image_dataset_from_directory`,
`shuffle=False` on val/test so predictions stay aligned with labels during evaluation.

**Caching is deliberately asymmetric** (see fix #9): `prefetch` only on train, `.cache()` on
val/test.

### Step 4 — `src/model.py`
`build_model(num_classes)` → one `tf.keras.Sequential([...])`:
`RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomContrast(0.1)` →
`Rescaling(1./127.5, offset=-1)` → `MobileNetV2(include_top=False, weights="imagenet")` with
`trainable=False` → `GlobalAveragePooling2D()` → `Dropout(0.2)` →
`Dense(num_classes, activation="softmax")`.

### Step 5 — `notebooks/train.ipynb` — the self-contained Colab notebook
Built, not executed. Ordered cells:

1. **GPU assert** — `tf.config.list_physical_devices('GPU')`; raise loudly with the
   Runtime → Change runtime type → T4 instruction if absent, rather than silently training
   on CPU for six hours.
2. **Get the archive** — two documented options: mount Drive (recommended — upload
   `archive.zip` to Drive once, no re-upload per session) or `files.upload()`. Unzips to
   **local `/content/`, never to Drive** — unzipping 20k small files onto mounted Drive is
   pathologically slow.
3. **Materialise `src/`** — `%%writefile` cells for `prepare_data.py`, `dataset.py`,
   `model.py`, mirroring the repo copies so the notebook needs no GitHub clone. *(Trade-off:
   these are duplicated. Training runs once, so drift risk is low; README notes to re-copy
   if the local modules change.)*
4. **Prepare data** — run the split, print the per-class table, print total counts.
5. **Load datasets, print `class_names` clearly.**
6. **Class weights** — `compute_class_weight("balanced", ...)` from train counts.
7. **Build + compile** — `adam`, `sparse_categorical_crossentropy`, `["accuracy"]`,
   plus `model.summary()`.
8. **Train** — `epochs=8`, `class_weight=`, `ModelCheckpoint` (best val accuracy) +
   `EarlyStopping(patience=3, restore_best_weights=True)`. Checkpoints to Drive so a
   disconnect doesn't lose the run.
9. **Training curves** — accuracy/loss plots, saved to `reports/`.
10. **Evaluate on the held-out test split, in Colab** — full `classification_report`,
    confusion matrix PNG, `metrics.json` (accuracy + macro F1 + per-class F1). These are the
    authoritative numbers for the report (fix #3).
11. **Sanity check** — `model.predict()` on a handful of test images, shown inline with
    predicted vs true labels.
12. **Save + package** — write `tomato_disease_mobilenetv2.keras`, `class_names.json`,
    `split_manifest.json`, and `reports/`; zip into one `artifacts.zip` for a single
    download. Print explicitly where each file goes in the repo.

### Step 6 — `src/advice.py`
`ADVICE: dict[str, {en, ur}]` for all 11 real class names from Step 0, plus
`get_advice(class_name)` with normalised lookup and an explicit fallback record.

### Step 7 — `src/infer.py`
`load_model(path)`, `load_class_names()`, and `predict(model, class_names, image, top_k=3)`
→ EXIF-transpose, RGB convert, resize 224×224, expand dims, `model.predict()`, return ranked
`[{class_name, confidence, advice}]`. **Signature stays stable — the UI depends on it.**

### Step 8 — `src/evaluate_model.py`
Local re-run of evaluation for convenience; same artifacts into `reports/`. Warns on any
class lacking an advice entry. Exits with a clear message if no checkpoint exists yet.
The Colab numbers remain authoritative.

### Step 9 — `src/streamlit_app.py`
Per the "field diagnostic scan" direction in `plan.md` — sage `#EEF1E7`, forest `#1E4635`,
ochre `#B8863B`; Fraunces + Work Sans + Noto Nastaliq Urdu; image left, ranked result cards
right with rank badge, gradient confidence bar, EN + RTL Urdu advice. CSS via
`st.markdown(..., unsafe_allow_html=True)`. Adds `@st.cache_resource`, a friendly "no model
yet — run the notebook first" state, and the low-confidence caveat.

### Step 10 — `README.md` + `CLAUDE.md`
Setup, exact command sequence, the Colab workflow, which split was used, where every number
came from, and honest limitations: no real-world field validation, frozen backbone only,
static advice text not agronomist-reviewed.

---

## Division of labour

**Claude Code builds:** every file above, including the notebook.

**You own:**
1. Downloading `archive.zip` from Kaggle *(done)*
2. Uploading it to Colab/Drive and running `notebooks/train.ipynb` on a T4
3. Sending back `artifacts.zip` → `models/` + `reports/`

*(Original manual step 4 — copying `class_names` back by hand — is removed by fix #5.)*

No simulated results, no invented accuracy numbers. Every figure in the README comes from a
real run you performed.

---

## Verification

1. `python src/prepare_data.py --src <extracted>` → per-class table; `data/train|val|test`
   populated; counts sum to the archive total; no filename appears in two splits.
2. `python -c "from src.dataset import get_datasets; ..."` → 11 class names, correct shapes.
3. `python -c "from src.model import build_model; build_model(11).summary()"` → output
   `(None, 11)`, MobileNetV2 params non-trainable.
4. Notebook runs end-to-end on T4 → `artifacts.zip` with checkpoint, `class_names.json`,
   `split_manifest.json`, `reports/`.
5. After unpacking: `python src/evaluate_model.py` → no missing-advice warnings, and local
   accuracy within ~1% of the Colab number. **A large gap means the splits diverged** —
   diff the two `split_manifest.json` files.
6. `streamlit run src/streamlit_app.py` from the repo root → theme applies (proves fix #4);
   upload a `data/test/` image; confirm top-3 cards, confidence bars, both languages render.
7. Negative check: upload a non-leaf photo → low-confidence caveat, not a confident
   wrong diagnosis.
