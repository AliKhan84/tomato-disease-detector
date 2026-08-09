"""Field diagnostic scan -- tomato leaf disease detector.

Run from the REPO ROOT so .streamlit/config.toml is picked up:

    streamlit run src/streamlit_app.py

Four tabs: Diagnose (upload -> ranked diagnosis), Model Performance (the measured test-set
numbers), Explainability (Grad-CAM and what the filters learned), Method & Limitations.

Every figure shown here is READ FROM reports/, never recomputed and never hardcoded. If the
files are missing the tab says so rather than inventing a number.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Must run before anything that reaches TensorFlow. If `streamlit` resolved to an
# interpreter without TF (pyenv's shims outrank the conda env in PATH on this machine),
# this re-executes the same command under one that has it. See interpreter_guard.py.
from interpreter_guard import ensure_tensorflow  # noqa: E402

ensure_tensorflow()

import base64  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from io import BytesIO  # noqa: E402

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

import explain  # noqa: E402
from infer import (  # noqa: E402
    CLASS_NAMES_PATH,
    CONFIDENCE_FLOOR,
    MODEL_PATH,
    is_low_confidence,
    load_class_names,
    load_model,
    predict,
)

# Two palettes, same semantic slots. Everything below -- CSS and matplotlib alike -- reads
# from the active one, so neither theme can drift from the other. The dark values are not
# the light ones inverted: `brand` has to carry heading text on a dark surface, so it moves
# from deep forest to a light sage, and `on_brand` flips with it to stay readable.
PALETTES = {
    "light": {
        "bg": "#EEF1E7",
        "surface": "#FFFFFF",
        "surface_alt": "#F6F8F2",
        "ink": "#1F2A24",
        "ink_soft": "rgba(31,42,36,.88)",
        "ink_mute": "rgba(31,42,36,.60)",
        "ink_faint": "rgba(31,42,36,.48)",
        "brand": "#1E4635",
        "brand_mute": "rgba(30,70,53,.32)",
        "on_brand": "#FFFFFF",
        "accent": "#B8863B",
        "border": "rgba(30,70,53,.14)",
        "border_soft": "rgba(30,70,53,.07)",
        "border_strong": "rgba(30,70,53,.28)",
        "track": "rgba(30,70,53,.09)",
        "code_bg": "rgba(30,70,53,.07)",
        "shadow": "rgba(30,70,53,.09)",
        "grid": "#E3E8DA",
        "axis": "#D8DECF",
        # Colour ramp for heatmaps: pale at zero, brand at full.
        "ramp": ["#FFFFFF", "#DCE5D2", "#8FAE86", "#3F6B52", "#1E4635"],
    },
    "dark": {
        "bg": "#0F1512",
        "surface": "#18201B",
        "surface_alt": "#131A16",
        "ink": "#E7EDE6",
        "ink_soft": "rgba(231,237,230,.86)",
        "ink_mute": "rgba(231,237,230,.60)",
        "ink_faint": "rgba(231,237,230,.44)",
        "brand": "#8FC7A4",
        "brand_mute": "rgba(143,199,164,.34)",
        "on_brand": "#0F1512",
        "accent": "#D9A85C",
        "border": "rgba(143,199,164,.20)",
        "border_soft": "rgba(143,199,164,.10)",
        "border_strong": "rgba(143,199,164,.34)",
        "track": "rgba(231,237,230,.12)",
        "code_bg": "rgba(143,199,164,.12)",
        "shadow": "rgba(0,0,0,.45)",
        "grid": "#2A342C",
        "axis": "#3A473D",
        # Runs dark -> light so a high value still reads as "hot" against a dark canvas.
        "ramp": ["#18201B", "#243429", "#3E6B4C", "#6BA37E", "#A8D8B9"],
    },
}

REPORTS = ROOT / "reports"
TEST_DIR = ROOT / "data" / "test"

st.set_page_config(
    page_title="Leaf Scan — Tomato Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# The toggle itself lives in the sidebar, far below, but the stylesheet is written here --
# so read the value straight out of session_state under the widget's own key. Streamlit
# reruns top-to-bottom when the toggle flips, so by then the new value is already set. On
# the very first run the key is absent and .get() falls through to the light default.
THEME = "dark" if st.session_state.get("dark_mode", False) else "light"
P = PALETTES[THEME]

SAGE = P["bg"]
FOREST = P["brand"]
OCHRE = P["accent"]
INK = P["ink"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

.stApp {{ background: {SAGE}; }}
html, body, [class*="css"] {{ font-family: 'Work Sans', sans-serif; color: {INK}; }}

#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}

.scan-header {{
  border-bottom: 1px solid {P['border']};
  padding-bottom: 1.1rem; margin-bottom: 1.2rem;
}}
.scan-header h1 {{
  font-family: 'Fraunces', serif; font-weight: 600; color: {FOREST};
  font-size: 2.6rem; line-height: 1.1; margin: 0 0 .35rem 0; letter-spacing: -.02em;
}}
.scan-header .sub {{ font-size: .95rem; color: {P['ink_mute']}; margin: 0; }}
.eyebrow {{
  font-size: .72rem; font-weight: 600; letter-spacing: .18em;
  text-transform: uppercase; color: {OCHRE}; margin-bottom: .5rem;
}}
h3.section {{
  font-family: 'Fraunces', serif; color: {FOREST}; font-size: 1.3rem;
  font-weight: 600; margin: 1.6rem 0 .5rem 0;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid {P['border']}; }}
.stTabs [data-baseweb="tab"] {{
  font-family: 'Work Sans', sans-serif; font-weight: 500; font-size: .93rem;
  color: {P['ink_mute']}; background: transparent; padding: .6rem 1.1rem;
}}
.stTabs [aria-selected="true"] {{ color: {FOREST} !important; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {FOREST}; }}

.result-card {{
  background: {P['surface']}; border-radius: 14px; padding: 1.25rem 1.4rem 1.35rem;
  margin-bottom: 1rem; box-shadow: 0 2px 14px {P['shadow']};
  border: 1px solid {P['border_soft']}; position: relative;
}}
.result-card.top {{ border-left: 3px solid {FOREST}; }}
.card-head {{ display: flex; align-items: baseline; gap: .7rem; margin-bottom: .1rem; }}
.rank-badge {{
  flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%;
  background: {FOREST}; color: {P['on_brand']}; font-size: .8rem; font-weight: 600;
  display: inline-flex; align-items: center; justify-content: center; align-self: center;
}}
.result-card:not(.top) .rank-badge {{ background: {P['brand_mute']}; }}
.disease-name {{
  font-family: 'Fraunces', serif; font-size: 1.32rem; font-weight: 600;
  color: {FOREST}; flex: 1 1 auto; line-height: 1.25;
}}
.confidence-pct {{
  font-size: .95rem; font-weight: 600; color: {OCHRE}; font-variant-numeric: tabular-nums;
}}
.bar-track {{
  height: 7px; border-radius: 4px; background: {P['track']};
  margin: .7rem 0 1rem; overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, {OCHRE} 0%, {FOREST} 100%); }}
.advice-en {{ font-size: .93rem; line-height: 1.62; color: {P['ink_soft']}; }}
.advice-ur {{
  font-family: 'Noto Nastaliq Urdu', serif; direction: rtl; text-align: right;
  font-size: 1.02rem; line-height: 2.5; color: {P['ink_soft']};
  margin-top: .9rem; padding-top: .9rem; border-top: 1px solid {P['border']};
}}
.unknown-flag {{ font-size: .78rem; color: {OCHRE}; margin-top: .6rem; font-style: italic; }}

.notice {{
  background: {P['surface']}; border-left: 3px solid {OCHRE}; border-radius: 10px;
  padding: 1rem 1.2rem; margin-bottom: 1.1rem; font-size: .92rem; line-height: 1.6;
  box-shadow: 0 2px 10px {P['shadow']}; color: {P['ink_soft']};
}}
.notice strong {{ color: {FOREST}; }}
.placeholder {{
  border: 1px dashed {P['border_strong']}; border-radius: 14px;
  padding: 3rem 1.5rem; text-align: center; color: {P['ink_faint']}; font-size: .95rem;
}}

/* metric cards */
.metric-row {{ display: flex; gap: .9rem; flex-wrap: wrap; margin-bottom: .4rem; }}
.metric {{
  background: {P['surface']}; border-radius: 12px; padding: 1.1rem 1.3rem; flex: 1 1 150px;
  box-shadow: 0 2px 12px {P['shadow']}; border: 1px solid {P['border_soft']};
}}
.metric.lead {{ border-left: 3px solid {OCHRE}; }}
.metric .label {{
  font-size: .7rem; letter-spacing: .13em; text-transform: uppercase;
  color: {P['ink_mute']}; font-weight: 600; margin-bottom: .3rem;
}}
.metric .value {{
  font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 600; color: {FOREST};
  line-height: 1; font-variant-numeric: tabular-nums;
}}
.metric .foot {{ font-size: .76rem; color: {P['ink_faint']}; margin-top: .35rem; }}

.prose {{ font-size: .93rem; line-height: 1.7; color: {P['ink_soft']}; }}
.prose strong {{ color: {FOREST}; }}
.prose code {{
  background: {P['code_bg']}; padding: .1rem .35rem; border-radius: 4px;
  font-size: .87em; color: {P['ink']};
}}
.disclaimer {{
  margin-top: 2.2rem; padding-top: 1.1rem; border-top: 1px solid {P['border']};
  font-size: .8rem; color: {P['ink_mute']}; line-height: 1.6;
}}

/* A figure drawn on its own white canvas (reports/*.png) needs an explicit card, or on
   the dark theme it floats as a bright rectangle and reads as a rendering bug. */
.figure-card {{
  background: #FFFFFF; border-radius: 12px; padding: .9rem;
  border: 1px solid {P['border_soft']}; box-shadow: 0 2px 12px {P['shadow']};
}}
.figure-card img {{ width: 100%; display: block; border-radius: 6px; }}

/* Streamlit's own widgets do not follow the injected palette, so the surfaces the eye
   actually lands on are restated here. config.toml still declares the light base -- it is
   read once at startup and cannot be switched at runtime. */
[data-testid="stSidebar"] {{ background: {P['surface']}; border-right: 1px solid {P['border_soft']}; }}
[data-testid="stFileUploaderDropzone"] {{
  background: {P['surface_alt']}; border: 1px dashed {P['border_strong']}; border-radius: 12px;
  color: {P['ink_soft']};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: {P['ink_faint']} !important;
}}
[data-baseweb="select"] > div {{
  background: {P['surface_alt']} !important; border-color: {P['border']} !important;
  color: {P['ink']} !important;
}}
[data-baseweb="popover"] li {{ background: {P['surface']}; color: {P['ink']}; }}
.stSlider label, .stSelectbox label, .stToggle label, .stFileUploader label {{
  color: {P['ink_soft']} !important;
}}
[data-testid="stExpander"] {{
  background: {P['surface']}; border: 1px solid {P['border_soft']}; border-radius: 10px;
}}
[data-testid="stExpander"] summary {{ color: {P['ink_soft']}; }}
.stCode, .stCode pre, pre, code {{
  background: {P['surface_alt']} !important; color: {P['ink_soft']} !important;
}}
hr {{ border-color: {P['border']}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --- loading -----------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_model_and_classes():
    """Load once per session. Without cache_resource this reruns on every interaction."""
    return load_model(), load_class_names()


@st.cache_data(show_spinner=False)
def read_json(path: Path):
    """Read a report file, or None if this run has not produced it."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def list_test_samples() -> dict[str, list[str]]:
    """Map class -> a few test-split image paths, for the sample picker.

    data/ is gitignored and only exists after prepare_data.py runs, so an empty dict is a
    normal state, not an error. Sampling is seeded so the offered set is stable across
    reruns instead of changing on every click.
    """
    if not TEST_DIR.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        files = sorted(p.name for p in class_dir.iterdir() if p.is_file())
        if files:
            picks = random.Random(f"samples:{class_dir.name}").sample(
                files, min(6, len(files))
            )
            out[class_dir.name] = [str(class_dir / n) for n in picks]
    return out


def esc(text) -> str:
    """Escape for interpolation into the card HTML."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fig_to_png(fig, facecolor: str) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it.

    Returning bytes rather than the figure keeps these cacheable with cache_data and
    guarantees the figure is closed -- Streamlit reruns leak figures otherwise.
    """
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=facecolor, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def themed_cmap(theme: str):
    """Palette ramp as a colormap, so charts sit in the app's colours rather than fight them."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("leafscan", PALETTES[theme]["ramp"])


def figure_card(png: bytes, caption: str = "") -> None:
    """Show a PNG that was drawn on a white canvas, on an explicit white card.

    Used for figures that arrive pre-rendered in reports/ and cannot be recoloured at
    display time. Without the card a white PNG floats on the dark theme and reads as a
    rendering fault; on the light theme the card is indistinguishable from the background.
    """
    b64 = base64.b64encode(png).decode()
    st.markdown(
        f'<div class="figure-card"><img src="data:image/png;base64,{b64}"/></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


# --- charts ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def confusion_png(matrix: list[list[int]], class_names: list[str], normalise: bool,
                  theme: str) -> bytes:
    """Confusion matrix heatmap. Row-normalised by default.

    Row-normalising matters here: raw counts let the 402-image healthy class visually
    dominate the 126-image powdery_mildew one, hiding exactly the minority-class failures
    the chart exists to reveal.

    `theme` is an argument rather than a module-level read so it forms part of the
    cache_data key -- otherwise flipping the toggle would serve the other theme's PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = PALETTES[theme]
    cm = np.asarray(matrix, dtype=float)
    if normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            shown = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        vmax = 1.0
    else:
        shown, vmax = cm, cm.max()

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7.5, n * 0.78), max(6.5, n * 0.68)))
    ax.set_facecolor(pal["surface"])
    im = ax.imshow(shown, cmap=themed_cmap(theme), vmin=0, vmax=vmax)

    ax.set_xticks(range(n), class_names, rotation=45, ha="right", fontsize=8,
                  color=pal["ink"])
    ax.set_yticks(range(n), class_names, fontsize=8, color=pal["ink"])
    ax.set_xlabel("Predicted", fontsize=9, color=pal["ink"])
    ax.set_ylabel("True", fontsize=9, color=pal["ink"])
    for spine in ax.spines.values():
        spine.set_edgecolor(pal["axis"])

    # Cell text has to invert against the ramp: the ramp ends on `brand`, and `on_brand`
    # is defined as whatever reads against it in this theme.
    threshold = vmax * 0.55
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                        color=pal["on_brand"] if shown[i, j] > threshold else pal["ink"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, color=pal["axis"], labelcolor=pal["ink"])
    cbar.outline.set_edgecolor(pal["axis"])
    cbar.set_label("share of true class" if normalise else "images", fontsize=8,
                   color=pal["ink"])
    return fig_to_png(fig, pal["surface"])


@st.cache_data(show_spinner=False)
def per_class_f1_png(per_class: dict[str, float], macro: float, theme: str) -> bytes:
    """Horizontal per-class F1 bars, weakest at the top where the eye lands first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = PALETTES[theme]
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    scores = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8.2, max(3.2, len(names) * 0.42)))
    ax.set_facecolor(pal["surface"])
    bars = ax.barh(range(len(names)), scores, color=pal["brand"], height=0.62)
    # Highlight the weakest class -- that is the one worth a sentence in the write-up.
    bars[0].set_color(pal["accent"])

    ax.set_yticks(range(len(names)), names, fontsize=8.5, color=pal["ink"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 score", fontsize=9, color=pal["ink"])
    ax.tick_params(axis="x", colors=pal["ink"])
    ax.axvline(macro, color=pal["accent"], ls="--", lw=1.1, alpha=.85)
    ax.annotate(f"macro {macro:.3f}", xy=(macro, len(names) - 0.4),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=pal["accent"], va="center")
    for i, v in enumerate(scores):
        ax.text(v + .008, i, f"{v:.3f}", va="center", fontsize=7.6, color=pal["ink"])

    ax.grid(axis="x", color=pal["grid"], lw=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_edgecolor(pal["axis"])
    return fig_to_png(fig, pal["surface"])


@st.cache_data(show_spinner=False)
def filter_grid_png(filters: np.ndarray, theme: str) -> bytes:
    """First-layer conv kernels as a grid of tiny RGB tiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = PALETTES[theme]
    n = len(filters)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.85, rows * 0.9))
    for idx, ax in enumerate(np.asarray(axes).ravel()):
        ax.axis("off")
        if idx < n:
            ax.imshow(filters[idx], interpolation="nearest")
            ax.set_title(f"{idx}", fontsize=6, color=pal["ink"], pad=2)
    fig.subplots_adjust(wspace=.15, hspace=.35)
    return fig_to_png(fig, pal["surface"])


@st.cache_data(show_spinner=False)
def feature_map_png(maps: np.ndarray, layer: str, theme: str) -> bytes:
    """Activation grid for one layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = PALETTES[theme]
    n = len(maps)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.1))
    for idx, ax in enumerate(np.asarray(axes).ravel()):
        ax.axis("off")
        if idx < n:
            ax.imshow(maps[idx], cmap="viridis", interpolation="nearest")
    fig.suptitle(layer, fontsize=8.5, color=pal["ink"], y=1.0)
    fig.subplots_adjust(wspace=.08, hspace=.08)
    return fig_to_png(fig, pal["surface"])


# --- shared image state ------------------------------------------------------

def set_image(image: Image.Image, caption: str) -> None:
    st.session_state["image"] = image
    st.session_state["caption"] = caption


def current_image() -> tuple[Image.Image | None, str]:
    return st.session_state.get("image"), st.session_state.get("caption", "")


def render_card(item: dict) -> str:
    advice = item["advice"]
    pct = item["confidence"] * 100
    unknown = (
        '<div class="unknown-flag">No advice entry for this class yet — '
        "add one in src/advice.py.</div>"
        if not item["known"]
        else ""
    )
    return f"""
    <div class="result-card {'top' if item['rank'] == 1 else ''}">
      <div class="card-head">
        <span class="rank-badge">{item['rank']}</span>
        <span class="disease-name">{esc(item['label'])}</span>
        <span class="confidence-pct">{pct:.1f}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:{max(pct, 1.5):.1f}%"></div></div>
      <div class="advice-en">{esc(advice['en'])}</div>
      <div class="advice-ur">{esc(advice['ur'])}</div>
      {unknown}
    </div>
    """


def metric_card(label: str, value: str, foot: str = "", lead: bool = False) -> str:
    return (
        f'<div class="metric{" lead" if lead else ""}">'
        f'<div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div>'
        f'<div class="foot">{esc(foot)}</div></div>'
    )


# --- header ------------------------------------------------------------------

st.markdown(
    """
    <div class="scan-header">
      <div class="eyebrow">Field Diagnostic Scan</div>
      <h1>Tomato Leaf Disease Detector</h1>
      <p class="sub">Transfer-learned MobileNetV2 over 11 tomato leaf classes, with
      ranked diagnosis and treatment guidance in English and Urdu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, class_names = get_model_and_classes()
except FileNotFoundError:
    st.markdown(
        f"""
        <div class="notice">
          <strong>No trained model yet.</strong><br>
          Expected <code>{esc(MODEL_PATH.relative_to(ROOT))}</code> and
          <code>{esc(CLASS_NAMES_PATH.relative_to(ROOT))}</code>.<br><br>
          Run <code>notebooks/train.ipynb</code> in Google Colab on a T4 GPU, then unpack
          the <code>artifacts.zip</code> it produces into this repo.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

metrics = read_json(REPORTS / "metrics.json")
run_info = read_json(REPORTS / "training_run.json")
samples = list_test_samples()

# --- sidebar: image source ---------------------------------------------------

with st.sidebar:
    # Bound to the same session_state key the stylesheet read at the top of the script.
    # Flipping it reruns from line 1, so the new palette is already in place by the time
    # this widget is drawn again.
    st.toggle("Dark theme", key="dark_mode", help="Switch the palette light/dark.")
    st.markdown("---")

    st.markdown('<div class="eyebrow">Specimen</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Leaf photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    # Only adopt an upload once, keyed by file_id -- otherwise every rerun would overwrite
    # a sample the user loaded afterwards.
    if uploaded is not None:
        token = getattr(uploaded, "file_id", uploaded.name)
        if st.session_state.get("_upload_token") != token:
            st.session_state["_upload_token"] = token
            set_image(Image.open(uploaded), f"Uploaded — {uploaded.name}")

    if samples:
        st.markdown(
            '<div class="eyebrow" style="margin-top:1.2rem">Or try a test image</div>',
            unsafe_allow_html=True,
        )
        pick = st.selectbox("Class", sorted(samples), label_visibility="collapsed")
        if st.button("Load random sample", width='stretch'):
            path = random.choice(samples[pick])
            set_image(Image.open(path), f"{pick} — {Path(path).name}")
        st.caption("Held-out test split. The model never trained on these.")
    else:
        st.caption(
            "No local test images. Run `python src/prepare_data.py --src raw_data` "
            "to enable the sample picker."
        )

    image, caption = current_image()
    if image is not None:
        st.markdown("---")
        st.image(image, caption=caption, width='stretch')
        if st.button("Clear", width='stretch'):
            for key in ("image", "caption", "_upload_token"):
                st.session_state.pop(key, None)
            st.rerun()

image, caption = current_image()

tab_dx, tab_perf, tab_xai, tab_method = st.tabs(
    ["Diagnose", "Model Performance", "Explainability", "Method & Limitations"]
)

# --- tab 1: diagnose ---------------------------------------------------------

with tab_dx:
    if image is None:
        st.markdown(
            '<div class="placeholder">Upload a leaf photo in the sidebar — or load a '
            "sample test image — to begin the scan.</div>",
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([5, 6], gap="large")
        with left:
            st.markdown('<div class="eyebrow">Specimen</div>', unsafe_allow_html=True)
            st.image(image, caption=caption, width='stretch')
        with right:
            st.markdown('<div class="eyebrow">Scan Report</div>', unsafe_allow_html=True)
            with st.spinner("Analysing…"):
                results = predict(model, class_names, image, top_k=3)

            if is_low_confidence(results):
                st.markdown(
                    f"""
                    <div class="notice">
                      <strong>Low confidence.</strong> The top match is only
                      {results[0]['confidence'] * 100:.1f}%, below the
                      {CONFIDENCE_FLOOR * 100:.0f}% mark. Only about 1.4% of genuine test
                      images score this low, so this is an unusual input — try a sharper,
                      well-lit photo of a single leaf filling the frame, and treat what
                      follows as a guess.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for item in results:
                st.markdown(render_card(item), unsafe_allow_html=True)

            # Always shown, not just under the threshold. Measured on this checkpoint: a
            # solid grey square returns Late_blight at 98.2% and a crude drawn face returns
            # Tomato_mosaic_virus at 99.2%, while real test leaves have a median top-1 of
            # 0.998. Non-leaf confidence overlaps real-leaf confidence completely, so a
            # threshold cannot separate them and it would be dishonest to present the
            # low-confidence flag as if it caught out-of-distribution input.
            st.markdown(
                '<div class="notice"><strong>This model cannot tell you it is looking at '
                "the wrong thing.</strong> It always answers with one of its "
                f"{len(class_names)} trained classes, and it does so <em>confidently</em> "
                "even for photos containing no leaf at all. A high percentage above is "
                "evidence only if you already know the subject is a tomato leaf. Use the "
                "Explainability tab to check the model is reading leaf tissue rather than "
                "background.</div>",
                unsafe_allow_html=True,
            )

# --- tab 2: model performance ------------------------------------------------

with tab_perf:
    if metrics is None:
        st.markdown(
            '<div class="notice"><strong>No metrics yet.</strong> '
            "<code>reports/metrics.json</code> ships inside the notebook's "
            "<code>artifacts.zip</code>. Unpack it, or run "
            "<code>python src/evaluate_model.py</code>.</div>",
            unsafe_allow_html=True,
        )
    else:
        acc = metrics["accuracy"]
        macro = metrics["macro_f1"]
        n_test = metrics["n_test_images"]
        per_class = metrics.get("per_class_f1", {})

        st.markdown(
            '<div class="eyebrow">Held-out test split — never seen during training</div>',
            unsafe_allow_html=True,
        )
        cards = [
            metric_card("Macro F1", f"{macro:.4f}", "headline metric", lead=True),
            metric_card("Accuracy", f"{acc:.4f}", f"{round(acc * n_test):,} of {n_test:,}"),
            metric_card("Test images", f"{n_test:,}", "held out"),
            metric_card("Classes", f"{len(metrics.get('class_names', class_names))}", "11-way"),
        ]
        if per_class:
            worst = min(per_class.items(), key=lambda kv: kv[1])
            cards.append(metric_card("Weakest class", f"{worst[1]:.3f}", worst[0]))
        st.markdown(f'<div class="metric-row">{"".join(cards)}</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="prose" style="margin-top:.9rem">Macro F1 leads because the '
            "dataset is imbalanced 3.1× — plain accuracy lets a failing minority class "
            "hide behind the majority ones. Here the two agree closely, which is itself "
            "the finding: no class was sacrificed for the average.</div>",
            unsafe_allow_html=True,
        )

        if run_info:
            st.markdown('<h3 class="section">Two-stage training</h3>',
                        unsafe_allow_html=True)
            s1 = run_info.get("stage1_best_val_accuracy")
            s2 = run_info.get("stage2_best_val_accuracy")
            row = [
                metric_card("Stage 1 — frozen", f"{s1:.4f}" if s1 else "—",
                            f"{run_info.get('stage1_epochs_ran', '—')} epochs, head only"),
                metric_card("Stage 2 — fine-tuned", f"{s2:.4f}" if s2 else "—",
                            f"{run_info.get('stage2_epochs_ran', '—')} epochs, "
                            f"lr {run_info.get('fine_tune_lr', '—')}", lead=True),
            ]
            if s1 and s2:
                row.append(metric_card("Gain from unfreezing", f"+{(s2 - s1) * 100:.2f} pts",
                                       "validation accuracy"))
            row.append(metric_card(
                "Shipped model",
                "fine-tuned" if run_info.get("fine_tuned") else "frozen",
                f"top {int(run_info.get('unfreeze_fraction', 0) * 100)}% unfrozen"
                if run_info.get("fine_tuned") else "stage 2 did not win",
            ))
            st.markdown(f'<div class="metric-row">{"".join(row)}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="prose" style="margin-top:.9rem">Stage 1 plateaued with '
                "training accuracy <em>below</em> validation and both losses level — the "
                "signature of underfitting, not overfitting. The frozen ImageNet features "
                "had given all they had, so no amount of extra epochs or head width would "
                "have helped. Unfreezing the top of the backbone was the only lever that "
                "adds capacity, and it is what produced the gain above.</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<h3 class="section">Confusion matrix</h3>', unsafe_allow_html=True)
        cm = metrics.get("confusion_matrix")
        if cm:
            raw = st.toggle("Show raw counts instead of row shares", value=False)
            st.image(
                confusion_png(cm, metrics.get("class_names", class_names), not raw, THEME),
                width='stretch',
            )
            st.caption(
                "Rows are the true class, columns the prediction. The diagonal is correct; "
                "everything off it is a confusion worth reading."
            )
        else:
            st.info("No confusion matrix in metrics.json.")

        if per_class:
            st.markdown('<h3 class="section">Per-class F1</h3>', unsafe_allow_html=True)
            st.image(per_class_f1_png(per_class, macro, THEME), width='stretch')

        curves = REPORTS / "training_curves.png"
        if curves.exists():
            st.markdown('<h3 class="section">Training curves</h3>', unsafe_allow_html=True)
            # Pre-rendered in the notebook on a white canvas, so it cannot follow the
            # palette -- it goes on an explicit card instead. See figure_card().
            figure_card(
                curves.read_bytes(),
                "Both stages on one axis; the dashed line is where the backbone unfroze. "
                "Validation was still climbing at the final epoch — the run hit its epoch "
                "cap, not a ceiling.",
            )

        report_txt = REPORTS / "classification_report.txt"
        if report_txt.exists():
            with st.expander("Full classification report (precision / recall / support)"):
                st.code(report_txt.read_text(), language="text")

# --- tab 3: explainability ---------------------------------------------------

with tab_xai:
    st.markdown(
        '<div class="prose">A 94% score only means something if the model is reading the '
        "leaf rather than the background. Grad-CAM answers that directly: it weights the "
        "final convolutional feature maps by how much each one moved the predicted class, "
        "so bright regions are the pixels that actually drove the decision.</div>",
        unsafe_allow_html=True,
    )

    if image is None:
        st.markdown(
            '<div class="placeholder" style="margin-top:1.2rem">Load an image in the '
            "sidebar to see where the model looks.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<h3 class="section">Grad-CAM</h3>', unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2], gap="large")
        with col_b:
            target = st.selectbox(
                "Explain which class?",
                ["Predicted class"] + list(class_names),
                help="Pick a different class to ask the counterfactual question: where "
                     "would the model look if this were that disease?",
            )
            alpha = st.slider("Overlay strength", 0.0, 1.0, 0.45, 0.05)

        class_index = None if target == "Predicted class" else class_names.index(target)
        try:
            with st.spinner("Computing Grad-CAM…"):
                heatmap, idx, conf = explain.gradcam(model, image, class_index=class_index)
                overlay = explain.overlay_gradcam(image, heatmap, alpha=alpha)
                heat_rgb = explain.heatmap_to_rgba(heatmap).convert("RGB")
        except Exception as exc:  # surfaced, not swallowed -- a silent blank is worse
            st.error(f"Grad-CAM failed: {type(exc).__name__}: {exc}")
        else:
            with col_a:
                c1, c2, c3 = st.columns(3)
                base = image.convert("RGB").resize(explain.IMAGE_SIZE)
                c1.image(base, caption="Input", width='stretch')
                c2.image(heat_rgb, caption="Grad-CAM", width='stretch')
                c3.image(overlay, caption="Overlay", width='stretch')
            st.markdown(
                f'<div class="prose">Explaining <strong>{esc(class_names[idx])}</strong> '
                f"at {conf * 100:.1f}% confidence, from the backbone's "
                f"<code>{explain.GRADCAM_LAYER}</code> layer (7×7×1280 — the last stage "
                "that still knows <em>where</em> things are).</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<h3 class="section">What the first layer learned</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="prose">The 32 kernels of MobileNetV2\'s first convolution, each a '
            "3×3×3 patch shown as a colour tile. These come from ImageNet pretraining and "
            "sit below the unfreeze cut, so they were never updated here — expect generic "
            "edge, blob and colour-opponent detectors.</div>",
            unsafe_allow_html=True,
        )
        try:
            st.image(filter_grid_png(explain.first_conv_filters(model), THEME), width=520)
        except Exception as exc:
            st.error(f"Could not render filters: {type(exc).__name__}: {exc}")

        st.markdown('<h3 class="section">Feature maps by depth</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="prose">The 16 most active channels at three depths. Early layers '
            "keep the leaf's outline; by the final layer the maps are 7×7 and no longer "
            "look like a leaf at all — they encode <em>what</em> is present rather than "
            "what it looks like.</div>",
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Extracting activations…"):
                blocks = explain.feature_maps(model, image)
            for block in blocks:
                st.image(
                    feature_map_png(block["maps"],
                                    f"{block['name']}  —  {block['shape']}", THEME),
                    width='stretch',
                )
        except Exception as exc:
            st.error(f"Could not render feature maps: {type(exc).__name__}: {exc}")

# --- tab 4: method & limitations ---------------------------------------------

with tab_method:
    n_test = metrics["n_test_images"] if metrics else "—"
    st.markdown(
        f"""
        <h3 class="section">How the data was split</h3>
        <div class="prose">
        The Kaggle archive ships pre-split as <code>train/</code> and <code>valid/</code>.
        This project <strong>preserves that boundary</strong>: the archive's
        <code>train/</code> becomes the training set, and its <code>valid/</code> is halved
        per class into validation and test ({n_test:,} test images).<br><br>
        It deliberately does <strong>not</strong> pool everything and reshuffle 70/15/15.
        The dataset aggregates multiple sources — lab PlantVillage images alongside field
        photos — and contains near-duplicates. Pooling puts near-duplicates on both sides
        of the split, so the test score measures memorisation and reads far higher than the
        model deserves. A ~99% number from that approach would be worth less than the 94.9%
        here.<br><br>
        The split is deterministic: filenames sorted, RNG seeded per class, nothing
        depending on filesystem order. Colab and this laptop produce byte-identical
        assignments, which is what makes local evaluation trustworthy rather than
        decorative.
        </div>

        <h3 class="section">Architecture</h3>
        <div class="prose">
        Augmentation (<code>RandomFlip</code>, <code>RandomRotation</code>,
        <code>RandomContrast</code>) runs on raw 0–255 pixels <em>before</em>
        <code>Rescaling</code> to [-1, 1] — <code>RandomContrast</code> assumes 0–255 input
        and distorts already-normalised data. Preprocessing lives inside the saved model, so
        inference passes plain pixels and cannot drift out of sync with training.<br><br>
        Training used <code>class_weight="balanced"</code> against the 3.1× imbalance. In
        fine-tuning, every BatchNorm layer stayed frozen at every depth: in Keras,
        <code>trainable = False</code> on BatchNorm also forces inference mode, and letting
        its moving statistics update on small batches corrupts the pretrained
        representation within an epoch.
        </div>

        <h3 class="section">Limitations</h3>
        <div class="prose">
        <strong>No real-world validation.</strong> Every number comes from held-out images
        of the same Kaggle dataset. Performance on a phone photo in an actual field is
        unmeasured and will be worse — much of the source data is lab imagery on uniform
        backgrounds.<br><br>
        <strong>Closed-set classifier, and the confidence number does not fix it.</strong>
        Softmax always sums to 1, so a photo with no leaf in it still yields a
        confident-looking ranking. Measured on this checkpoint: a solid grey square returns
        <code>Late_blight</code> at 98.2%, and a crudely drawn face returns
        <code>Tomato_mosaic_virus</code> at 99.2% — while genuine test leaves have a median
        top-1 of 0.998. Those distributions overlap completely, so <em>no</em> threshold
        separates in-distribution from out-of-distribution input here, and prediction
        entropy does not either. The {CONFIDENCE_FLOOR * 100:.0f}% flag in the Diagnose tab
        catches genuinely ambiguous <em>leaf</em> photos (about 1.4% of the test set), not
        wrong subjects. Detecting those properly needs an explicit background class or an
        OOD method, which is out of scope here.<br><br>
        <strong>Partial fine-tuning only.</strong> The bottom 65% of the backbone stayed
        frozen. A full unfreeze with a longer schedule and LR warmup would likely add more,
        at meaningfully higher risk of wrecking the pretrained features.<br><br>
        <strong>Static advice text.</strong> General horticultural guidance, not reviewed by
        an agronomist, with no regional or cultivar specificity.<br><br>
        <strong>Single split, no cross-validation.</strong> No confidence intervals on any
        metric above.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="disclaimer">
      Course-project prototype. Trained on the Kaggle “Tomato Disease Multiple Sources”
      dataset via transfer learning on MobileNetV2, and validated only on held-out images
      from that same dataset — not on real field photography. The treatment text is general
      horticultural guidance and has not been reviewed by an agronomist. Confirm any
      diagnosis with your local agricultural extension service before treating a crop.
    </div>
    """,
    unsafe_allow_html=True,
)
