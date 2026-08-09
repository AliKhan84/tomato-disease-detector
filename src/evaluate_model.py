"""Evaluate the trained checkpoint on the held-out test split.

Writes reports/classification_report.txt, reports/confusion_matrix.png and
reports/metrics.json.

The authoritative numbers for the write-up are the ones the Colab notebook produced,
since Colab is where training and the canonical split happened. This script is for
re-checking locally -- if its accuracy differs from the Colab number by more than a
point or so, the two machines built different splits; diff data/split_manifest.json
against the copy in artifacts.zip.

Usage:  python src/evaluate_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advice import missing_advice  # noqa: E402
from dataset import get_datasets  # noqa: E402
from infer import CLASS_NAMES_PATH, MODEL_PATH, load_class_names, load_model  # noqa: E402


def collect_labels(dataset) -> np.ndarray:
    """True labels in dataset order. Valid only because test_ds has shuffle=False."""
    return np.concatenate([y.numpy() for _, y in dataset])


def plot_confusion(cm: np.ndarray, class_names: list[str], dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # Row-normalise so large classes do not visually swamp small ones.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.85), max(7, n * 0.75)))
    im = ax.imshow(norm, cmap="YlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")

    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "#1E4635",
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=CLASS_NAMES_PATH)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    try:
        model = load_model(args.model)
        class_names = load_class_names(args.class_names)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    _, _, test_ds, ds_class_names = get_datasets(args.data_dir)

    if ds_class_names != class_names:
        print(
            "WARNING: class order differs between models/class_names.json and the local\n"
            "data/ folders. Metrics below would be mislabelled, so stopping.\n"
            f"  checkpoint: {class_names}\n  local data: {ds_class_names}"
        )
        return 1

    absent = missing_advice(class_names)
    if absent:
        print(
            f"WARNING: no advice entry for {absent}. These will show a placeholder in the\n"
            "UI. Add them to src/advice.py.\n"
        )

    print(f"Evaluating on {args.data_dir / 'test'} ...")
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = collect_labels(test_ds)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    # Macro F1 leads: this dataset is imbalanced, and plain accuracy lets a failing
    # minority class hide behind the majority ones.
    print(f"\nMacro F1 : {macro_f1:.4f}   <- headline metric")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Test images: {len(y_true)}\n")
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classification_report.txt").write_text(
        f"Macro F1: {macro_f1:.4f}\nAccuracy: {accuracy:.4f}\n"
        f"Test images: {len(y_true)}\n\n{report}"
    )
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "n_test_images": int(len(y_true)),
                "class_names": class_names,
                "per_class_f1": {
                    name: float(score) for name, score in zip(class_names, per_class_f1)
                },
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        )
    )
    plot_confusion(cm, class_names, args.out / "confusion_matrix.png")

    print(f"Wrote classification_report.txt, metrics.json, confusion_matrix.png to {args.out}")

    worst = min(zip(class_names, per_class_f1), key=lambda pair: pair[1])
    print(f"Weakest class: {worst[0]} (F1 {worst[1]:.3f}) -- worth a sentence in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
