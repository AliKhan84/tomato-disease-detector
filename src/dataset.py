"""Dataset loading for train / val / test.

Augmentation deliberately does NOT live here -- it is inside the model (see model.py),
so the saved .keras checkpoint carries its own preprocessing and inference code does not
have to reimplement it.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def get_datasets(
    data_dir: str | Path = ROOT / "data",
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """Return (train_ds, val_ds, test_ds, class_names).

    class_names comes from image_dataset_from_directory, which sorts alphabetically.
    That ordering is the contract between training and inference -- it is saved to
    models/class_names.json by the notebook and read back by infer.py.
    """
    data_dir = Path(data_dir)
    missing = [s for s in ("train", "val", "test") if not (data_dir / s).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing split(s) {missing} under {data_dir}.\n"
            "Run:  python src/prepare_data.py --src <extracted-archive>"
        )

    def load(split: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            data_dir / split,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",  # pairs with sparse_categorical_crossentropy
            # val/test must stay in a fixed order so predictions line up with the
            # labels collected during evaluation.
            shuffle=shuffle,
            seed=42 if shuffle else None,
        )

    train_ds = load("train", shuffle=True)
    val_ds = load("val", shuffle=False)
    test_ds = load("test", shuffle=False)
    class_names = list(train_ds.class_names)

    for name, ds in (("val", val_ds), ("test", test_ds)):
        if list(ds.class_names) != class_names:
            raise SystemExit(
                f"Class mismatch between train and {name}:\n"
                f"  train: {class_names}\n  {name}: {list(ds.class_names)}\n"
                "Every split needs the same class folders. Re-run prepare_data.py --force."
            )

    # Caching is asymmetric on purpose. image_dataset_from_directory yields decoded
    # float32, so an in-memory .cache() on ~20k training images costs roughly
    # 20000 * 224 * 224 * 3 * 4 bytes ~= 12 GB -- more than a free Colab T4 instance
    # has (~12.7 GB), and the session dies mid-epoch-1 with "your runtime has crashed".
    # Do not add .cache() to train here. If epochs turn out to be I/O-bound, use a
    # disk cache -- .cache(filename="/content/tf_cache_train") -- not a memory one.
    # val/test are ~10x smaller and cache safely.
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def count_per_class(data_dir: str | Path, split: str, class_names: list[str]) -> list[int]:
    """Image count per class, in class_names order. Used for class weights."""
    root = Path(data_dir) / split
    return [
        sum(1 for p in (root / name).iterdir() if p.is_file()) for name in class_names
    ]
