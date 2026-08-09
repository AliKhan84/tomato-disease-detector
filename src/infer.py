"""Model loading and single-image prediction.

The predict() signature is a contract with streamlit_app.py -- keep it stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # tolerate being imported as a module or as a top-level script
    from .advice import get_advice
except ImportError:  # pragma: no cover
    from advice import get_advice

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "tomato_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)

# Below this top-1 probability, treat the result as unreliable. Softmax always sums to 1,
# so a photo with no leaf in it still produces a confident-looking winner -- the UI needs
# to be able to say "this doesn't look like something I was trained on".
CONFIDENCE_FLOOR = 0.50


def load_model(path: str | Path = MODEL_PATH):
    """Load the trained checkpoint. Raises FileNotFoundError with next steps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}.\n"
            "Train it first: run notebooks/train.ipynb in Google Colab, then unpack the\n"
            "resulting artifacts.zip so models/ contains the .keras file and "
            "class_names.json."
        )
    import tensorflow as tf  # imported lazily; keeps CLI --help fast

    # Preprocessing (augmentation + Rescaling) is baked into the saved model, so nothing
    # needs to be reapplied here.
    return tf.keras.models.load_model(path)


def load_class_names(path: str | Path = CLASS_NAMES_PATH) -> list[str]:
    """Read the class order saved alongside the checkpoint.

    This file is the contract that removes the old manual "copy class_names out of Colab
    and hand-edit advice.py" step. Order matters: it matches the model's output columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No class names at {path}.\n"
            "It is written by notebooks/train.ipynb and ships inside artifacts.zip."
        )
    data = json.loads(path.read_text())
    names = data["class_names"] if isinstance(data, dict) else data
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} does not contain a non-empty list of class names.")
    return list(names)


def preprocess(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """PIL image -> (1, H, W, 3) float32 array in 0-255.

    exif_transpose first: phone cameras store orientation in EXIF rather than rotating
    pixels, so a portrait photo arrives sideways without it. convert("RGB") then handles
    RGBA screenshots, greyscale scans, and palettised PNGs, all of which would otherwise
    produce the wrong channel count.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB").resize(image_size, Image.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype="float32"), axis=0)


def predict(model, class_names: list[str], image: Image.Image, top_k: int = 3) -> list[dict]:
    """Classify one PIL image.

    Returns up to top_k dicts, highest confidence first:
        {rank, class_name, label, confidence, advice, known}
    The final Dense layer already applies softmax, so probs need no further activation.
    """
    probs = model.predict(preprocess(image), verbose=0)[0]

    if len(probs) != len(class_names):
        raise ValueError(
            f"Model outputs {len(probs)} classes but class_names has "
            f"{len(class_names)}. models/class_names.json does not match this checkpoint "
            "-- re-download both from the same Colab run."
        )

    order = np.argsort(probs)[::-1][: max(1, top_k)]
    results = []
    for rank, idx in enumerate(order, start=1):
        class_name = class_names[int(idx)]
        advice = get_advice(class_name)
        results.append(
            {
                "rank": rank,
                "class_name": class_name,
                "label": advice["label"],
                "confidence": float(probs[int(idx)]),
                "advice": advice,
                "known": advice["known"],
            }
        )
    return results


def is_low_confidence(results: list[dict], floor: float = CONFIDENCE_FLOOR) -> bool:
    """True when the top prediction is too weak to present as a diagnosis."""
    return not results or results[0]["confidence"] < floor
