
# Crop Leaf Disease Detector — Project Plan for Claude Code

## What this is
A tomato leaf disease classifier (MVP) for an Applied AI & ML course
midterm. Transfer-learning CNN (MobileNetV2 base, Keras `Sequential`
model, frozen base) served through a Streamlit app with a
custom-designed UI. 2-day build. Nothing in this repo exists yet —
build all of it from this plan.

## Tech stack
**TensorFlow / Keras — `Sequential` model API.** This matches what the
course actually teaches, so use it consistently: `tf.keras.Sequential`,
`model.compile()` / `model.fit()`, not a manual training loop, and not
PyTorch. `tf.keras.applications.MobileNetV2` as the pretrained base.

## Read this first — who does what

**Claude Code owns:** building everything below from scratch — data
pipeline, model, training notebook, evaluation script, advice
dictionary, inference module, and the Streamlit UI.

**The human owns these manual steps. Do not attempt them, and do not
report having done them:**
1. Downloading the dataset from Kaggle (link below) — this environment
   has no Kaggle access
2. Opening and running `notebooks/train.ipynb` in Google Colab — needs
   GPU, which this environment doesn't have
3. Downloading the resulting `.keras` checkpoint from Colab into
   `models/tomato_disease_mobilenetv2.keras` in this repo
4. Reporting back the `class_names` list the notebook prints, so
   `advice.py`'s dictionary keys can be corrected to match

If asked to "train the model" or "get the dataset," say plainly that
these are manual/Colab steps outside this environment, and point back
to this file. Don't simulate results or invent accuracy numbers.

## Dataset
**Kaggle — Tomato Disease Multiple Sources**
https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources

~20,000 images, 11 classes (10 diseases + healthy), mix of lab and
real-field photos. Human downloads this manually, or via the Kaggle API
with their own `kaggle.json` credentials inside Colab.

## Repo structure to build

```
crop-disease-mvp/
├── README.md
├── requirements.txt
├── data/                       # gitignored — created by prepare_data.py
├── models/                     # gitignored — holds the trained .keras checkpoint
├── notebooks/
│   └── train.ipynb             # training notebook, run in Google Colab
└── src/
    ├── prepare_data.py
    ├── dataset.py
    ├── model.py
    ├── evaluate_model.py
    ├── advice.py
    ├── infer.py
    ├── streamlit_app.py
    └── .streamlit/
        └── config.toml
```

## Build tasks, in order

1. **Data pipeline** — `src/prepare_data.py`: splits a raw Kaggle
   download (`raw_data/<class>/*.jpg`) into `data/train|val|test`
   (70/15/15 split). Framework-agnostic — just file copying. Check the
   actual downloaded folder structure before assuming it's flat
   per-class folders — adjust if not.

2. **Dataset loading** — `src/dataset.py`: use
   `tf.keras.utils.image_dataset_from_directory` for each of
   `data/train`, `data/val`, `data/test` (image_size=(224, 224),
   batch_size=32). Expose a `get_datasets(data_dir="data")` function
   returning `(train_ds, val_ds, test_ds, class_names)`. Don't hand-roll
   augmentation here — it belongs inside the model itself (see below).

3. **Model definition** — `src/model.py`: a `build_model(num_classes)`
   function returning a `tf.keras.Sequential` model:
   - A small augmentation block (also `Sequential`, or inlined as
     layers): `RandomFlip("horizontal")`, `RandomRotation(0.1)`,
     `RandomContrast(0.1)` — real phone photos vary in angle and
     lighting, this is what keeps the model from only working on
     dataset-clean images
   - `Rescaling` layer matching MobileNetV2's expected [-1, 1] input
     range (equivalent to
     `tf.keras.applications.mobilenet_v2.preprocess_input`), so
     preprocessing lives inside the saved model rather than needing to
     be reimplemented in `infer.py`
   - `tf.keras.applications.MobileNetV2(include_top=False,
     weights="imagenet")` as the base, with `base_model.trainable =
     False` — frozen backbone, matching the 2-day time budget (no
     fine-tuning)
   - `GlobalAveragePooling2D()`, `Dropout(0.2)`, then a final
     `Dense(num_classes, activation="softmax")`
   All of this stacks directly into one `Sequential([...])` call — the
   base model counts as a single layer in the stack.

4. **Training notebook** — `notebooks/train.ipynb` (a Jupyter notebook,
   not a `.py` script — this runs in Google Colab). Structure it as:
   - Cell 1: install/check deps (`pip install -q tensorflow` if needed
     — Colab usually has it preinstalled)
   - Cell 2: add repo root to `sys.path`, import `get_datasets` from
     `src/dataset.py` and `build_model` from `src/model.py` — don't
     duplicate that code inline, keep one source of truth shared with
     local inference/eval code
   - Cell 3: load datasets, **print `class_names` clearly** (the human
     needs to copy this out for `advice.py`)
   - Cell 4: build the model, `model.compile(optimizer="adam",
     loss="sparse_categorical_crossentropy", metrics=["accuracy"])`
   - Cell 5: `model.fit(train_ds, validation_data=val_ds, epochs=8,
     callbacks=[ModelCheckpoint(...)])` — checkpoint on best val
     accuracy
   - Cell 6: save/confirm the final model at
     `models/tomato_disease_mobilenetv2.keras`
   - Cell 7: a quick sanity-check cell running `model.predict()` on a
     couple of validation images inline, so the human can eyeball that
     predictions look reasonable before leaving Colab
   Build this notebook, do not attempt to execute it.

5. **Evaluation** — `src/evaluate_model.py`: load the model with
   `tf.keras.models.load_model()`, run predictions over `test_ds`,
   collect true labels, and print a confusion matrix + per-class
   precision/recall/F1 via `sklearn.metrics.classification_report` /
   `confusion_matrix`. Runs only once a real checkpoint exists in
   `models/`.

6. **Advice dictionary** — `src/advice.py`: static
   `class_name -> {en, ur}` treatment text. Use placeholder keys based
   on common tomato-disease naming conventions (Early_blight,
   Late_blight, Bacterial_spot, Septoria_leaf_spot, Leaf_Mold,
   Target_Spot, Spider_mites, Yellow_Leaf_Curl_Virus, Mosaic_virus,
   Healthy) and flag clearly in a comment that these must be reconciled
   against the notebook's real `class_names` output — a mismatch means
   predictions silently show no advice text, not an error.

7. **Inference** — `src/infer.py`: `load_model(path)` wrapping
   `tf.keras.models.load_model()`, and `predict(model, class_names,
   image, top_k=3)` that resizes the PIL image to 224×224, expands
   dims, runs `model.predict()` (softmax already applied by the final
   layer), and returns the top-k classes with confidence + advice text
   looked up from `advice.py`. Keep this function signature stable —
   the Streamlit UI depends on it.

8. **UI** — `src/streamlit_app.py`: build per the design direction
   below, calling `infer.py`'s `load_model()` / `predict()`.

## Design direction for the UI
"Field diagnostic scan" concept — not a generic dashboard template.

- **Palette:** sage-green background `#EEF1E7`, forest-green primary
  `#1E4635`, ochre accent `#B8863B`, white result cards with a soft
  shadow
- **Type:** Fraunces (serif, headers/disease names) + Work Sans (body) +
  Noto Nastaliq Urdu (Urdu advice text, right-aligned, RTL)
- **Layout:** uploaded leaf image on the left, a "scan report" of
  ranked result cards on the right — each card has a numbered rank
  badge, a gradient confidence bar (ochre → forest green), and
  English + Urdu advice stacked
- Implement via CSS injected into Streamlit (`st.markdown(...,
  unsafe_allow_html=True)`) plus `src/.streamlit/config.toml` for base
  theming — not a separate frontend framework, no time budget for that
  in a 2-day build

## What "done" looks like
- `streamlit run streamlit_app.py` runs cleanly against a real
  `.keras` checkpoint, no errors
- Uploading a held-out test image returns top-3 predictions with
  confidence bars and bilingual advice text, styled per the design
  direction above
- `evaluate_model.py` output (accuracy + confusion matrix) is saved
  somewhere the report can reference
- README documents setup, the manual Colab steps, and an honest
  limitations section: no real-world validation, frozen-backbone only,
  static advice text

## requirements.txt should include
`tensorflow`, `streamlit`, `pillow`, `scikit-learn` — no `torch` /
`torchvision`.

## Note on filename
This file is named `plan.md`. If running this with Claude Code
specifically, consider also saving a copy as `CLAUDE.md` at the repo
root — Claude Code auto-loads that filename as project context without
being told to read it.
