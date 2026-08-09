"""Visual explanations for the trained classifier: Grad-CAM, conv filters, feature maps.

WHY THIS FILE LOOKS THE WAY IT DOES
-----------------------------------
The saved model is a Sequential whose fifth layer is the whole MobileNetV2 as a *nested*
Functional model:

    RandomFlip -> RandomRotation -> RandomContrast -> Rescaling
      -> mobilenetv2_1.00_224 (154 layers)
      -> GlobalAveragePooling2D -> Dropout -> Dense(11, softmax)

The textbook Grad-CAM recipe builds `Model(model.input, [conv_layer.output, model.output])`.
That does NOT work here: `out_relu.output` is a symbolic tensor belonging to the *inner*
graph, and the outer Sequential never sees it, so Keras 3 raises a graph-disconnected
error. Rather than fight it, split_model() cuts the network into three segments and
forward_from_backbone() re-runs them by hand. That is exact -- same layers, same order,
same weights -- and survives the nesting.

Every layer is called with training=False. The three augmentation layers are no-ops at
inference, and Rescaling stays active, so these functions take plain 0-255 pixels exactly
like infer.predict() does. Do not pre-normalise before calling in here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:  # tolerate module or top-level-script import, same as infer.py
    from .infer import preprocess, IMAGE_SIZE
except ImportError:  # pragma: no cover
    from infer import preprocess, IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]

# Final activation of the backbone: 7x7x1280. The last spatial map before global pooling
# throws the geometry away, which is exactly what Grad-CAM needs.
GRADCAM_LAYER = "out_relu"

# Three depths that tell the "edges -> texture -> semantics" story. Resolved against the
# real model at call time; anything missing is skipped rather than raising.
FEATURE_MAP_LAYERS = ("block_1_expand_relu", "block_6_expand_relu", "out_relu")


def find_backbone(model):
    """Return the nested MobileNetV2 inside the Sequential.

    Identified structurally (the one layer that itself has .layers) rather than by name or
    index, so it keeps working if the stack around it changes.
    """
    nested = [layer for layer in model.layers if hasattr(layer, "layers")]
    if len(nested) != 1:
        raise ValueError(
            f"Expected exactly one nested backbone model, found {len(nested)}: "
            f"{[l.name for l in nested]}. explain.py assumes the architecture built by "
            "src/model.py."
        )
    return nested[0]


def split_model(model):
    """Cut the Sequential into (pre_layers, backbone, head_layers).

    pre_layers  -- augmentation + Rescaling, everything before the backbone
    backbone    -- the nested MobileNetV2
    head_layers -- pooling, dropout, classifier
    """
    backbone = find_backbone(model)
    index = model.layers.index(backbone)
    return model.layers[:index], backbone, model.layers[index + 1 :]


def apply_preprocessing(model, x):
    """Run the outer preprocessing layers (augmentation no-ops + Rescaling) on 0-255 input."""
    pre, _, _ = split_model(model)
    for layer in pre:
        x = layer(x, training=False)
    return x


def forward_from_backbone(model, conv_out):
    """Run the classifier head on a backbone activation tensor."""
    _, _, head = split_model(model)
    y = conv_out
    for layer in head:
        y = layer(y, training=False)
    return y


def gradcam(model, image: Image.Image, class_index: int | None = None,
            layer_name: str = GRADCAM_LAYER):
    """Grad-CAM heatmap for one PIL image.

    Returns (heatmap, class_index, confidence) where heatmap is a float32 array in [0, 1]
    at the conv layer's spatial resolution (7x7 for out_relu) -- callers resize it.

    class_index defaults to the model's own top prediction. Pass an explicit index to ask
    the counterfactual question "where would you look if this were class k?".
    """
    import tensorflow as tf

    _, backbone, _ = split_model(model)
    try:
        target_layer = backbone.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Layer {layer_name!r} not found in backbone {backbone.name!r}."
        ) from exc

    # Sub-model: backbone input -> chosen activation. Built from the backbone's OWN graph,
    # which is self-contained, so this is the one piece of graph surgery that is safe.
    feature_model = tf.keras.Model(backbone.inputs, target_layer.output)

    x = tf.convert_to_tensor(preprocess(image))
    x = apply_preprocessing(model, x)

    with tf.GradientTape() as tape:
        conv_out = feature_model(x, training=False)
        # watch() before conv_out is consumed below, or the tape has no path to it.
        tape.watch(conv_out)
        preds = forward_from_backbone(model, conv_out)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:  # pragma: no cover -- would mean the tape path broke
        raise RuntimeError(
            "Grad-CAM got no gradient. The forward pass through the head did not connect "
            "to the watched activation."
        )

    # Channel importance = spatially averaged gradient; heatmap = that-weighted channel sum.
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)

    # ReLU: only evidence *for* the class matters. Negative regions argue against it and
    # would otherwise show up as spurious hot spots once normalised.
    heatmap = tf.nn.relu(heatmap)
    peak = tf.reduce_max(heatmap)
    # A dead-flat map (peak 0) means no positive evidence anywhere; return zeros rather
    # than dividing by zero and producing NaNs.
    heatmap = heatmap / peak if peak > 0 else heatmap

    return heatmap.numpy().astype("float32"), int(class_index), float(preds[0][class_index])


def heatmap_to_rgba(heatmap: np.ndarray, size: tuple[int, int] = IMAGE_SIZE,
                    cmap_name: str = "inferno") -> Image.Image:
    """Upscale a small heatmap and colourise it. Bicubic keeps 7x7 from looking like Lego."""
    import matplotlib

    # matplotlib.colormaps is the 3.5+ API; cm.get_cmap was removed in 3.9.
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except AttributeError:  # pragma: no cover -- matplotlib < 3.5
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)

    small = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype("uint8"), mode="L")
    upscaled = np.asarray(small.resize(size, Image.BICUBIC), dtype="float32") / 255.0
    return Image.fromarray((cmap(upscaled) * 255).astype("uint8"), mode="RGBA")


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45,
                    cmap_name: str = "inferno") -> Image.Image:
    """Blend a Grad-CAM heatmap over the original image at display resolution."""
    from PIL import ImageOps

    base = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    coloured = heatmap_to_rgba(heatmap, IMAGE_SIZE, cmap_name).convert("RGB")
    return Image.blend(base, coloured, alpha)


def first_conv_filters(model, max_filters: int = 32) -> np.ndarray:
    """First-layer conv kernels as (n, 3, 3, 3), each normalised to [0, 1] for display.

    MobileNetV2's Conv1 is (3, 3, 3, 32): 32 filters over RGB, so each one is directly
    viewable as a tiny colour image. Deeper kernels have channel counts that do not map to
    RGB and are not worth rendering this way.
    """
    _, backbone, _ = split_model(model)
    import tensorflow as tf

    conv = next(l for l in backbone.layers if isinstance(l, tf.keras.layers.Conv2D))
    kernels = conv.kernel.numpy()  # (h, w, in_ch, out_ch)
    kernels = np.transpose(kernels, (3, 0, 1, 2))[:max_filters]  # -> (n, h, w, in_ch)

    # Per-filter min-max. A global normalisation would let one high-contrast kernel
    # flatten all the others into grey mush.
    flat = kernels.reshape(len(kernels), -1)
    lo = flat.min(axis=1).reshape(-1, 1, 1, 1)
    hi = flat.max(axis=1).reshape(-1, 1, 1, 1)
    return ((kernels - lo) / np.maximum(hi - lo, 1e-8)).astype("float32")


def feature_maps(model, image: Image.Image, layer_names=FEATURE_MAP_LAYERS,
                 max_channels: int = 16) -> list[dict]:
    """Activations at several depths for one image.

    Returns [{name, shape, maps}], maps being (n, h, w) normalised to [0, 1]. Layers that
    do not exist in this backbone are skipped silently -- the names are a MobileNetV2
    convention, not a guarantee.
    """
    import tensorflow as tf

    _, backbone, _ = split_model(model)
    available = {l.name for l in backbone.layers}
    wanted = [n for n in layer_names if n in available]
    if not wanted:
        return []

    extractor = tf.keras.Model(
        backbone.inputs, [backbone.get_layer(n).output for n in wanted]
    )
    x = apply_preprocessing(model, tf.convert_to_tensor(preprocess(image)))
    outputs = extractor(x, training=False)
    if not isinstance(outputs, list):
        outputs = [outputs]

    results = []
    for name, activation in zip(wanted, outputs):
        act = activation.numpy()[0]  # (h, w, channels)
        # Busiest channels first: a channel that never fires is a black square that tells
        # the reader nothing.
        energy = act.reshape(-1, act.shape[-1]).mean(axis=0)
        picks = np.argsort(energy)[::-1][:max_channels]
        maps = np.transpose(act[:, :, picks], (2, 0, 1))
        lo = maps.min(axis=(1, 2), keepdims=True)
        hi = maps.max(axis=(1, 2), keepdims=True)
        results.append({
            "name": name,
            "shape": tuple(act.shape),
            "maps": ((maps - lo) / np.maximum(hi - lo, 1e-8)).astype("float32"),
        })
    return results
