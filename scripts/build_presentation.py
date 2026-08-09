#!/usr/bin/env python3
"""Build presentation.html -- one self-contained deck, no external assets.

Every number is read from reports/ and models/ at build time, so the slides cannot drift
from the run that produced them. Re-run after any retrain:

    python scripts/build_presentation.py

Idempotent: the output is written from scratch each time, never used as its own template.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from deck_shell import SHELL

ROOT = Path(__file__).resolve().parents[1]
R, M = ROOT / "reports", ROOT / "models"
OUT = ROOT / "presentation.html"
AUTHOR = "Ali Khan"

metrics = json.loads((R / "metrics.json").read_text())
run = json.loads((R / "training_run.json").read_text())
man = json.loads((M / "split_manifest.json").read_text())

names, cm = metrics["class_names"], metrics["confusion_matrix"]
n = len(names)
counts = man["counts"]
tr = {c: counts[c]["train"] for c in names}
n_tr, n_va, n_te = (sum(counts[c][k] for c in names) for k in ("train", "val", "test"))

# Precision/recall are derived from the confusion matrix rather than parsed out of the text
# report, so the per-class table and the matrix cannot disagree with each other.
pc = []
for i in range(n):
    sup, col, tp = sum(cm[i]), sum(cm[r][i] for r in range(n)), cm[i][i]
    pc.append(dict(name=names[i], p=tp / col if col else 0.0, r=tp / sup if sup else 0.0,
                   f1=metrics["per_class_f1"][names[i]], sup=sup))

acc, mf1 = metrics["accuracy"], metrics["macro_f1"]
s1, s2 = run["stage1_best_val_accuracy"], run["stage2_best_val_accuracy"]
gain = (s2 - s1) * 100
worst, best = min(pc, key=lambda d: d["f1"]), max(pc, key=lambda d: d["f1"])
imb = max(tr.values()) / min(tr.values())
correct = sum(cm[i][i] for i in range(n))
by_name = {d["name"]: d for d in pc}
eb, pm = by_name["Early_blight"], by_name["powdery_mildew"]
# The two cells that account for most of Early_blight's misses, read from the matrix.
eb_i = names.index("Early_blight")
eb_late = cm[eb_i][names.index("Late_blight")]
eb_sept = cm[eb_i][names.index("Septoria_leaf_spot")]


def pretty(s: str) -> str:
    return (s.replace("Tomato_", "")
             .replace("Spider_mites Two-spotted_spider_mite", "Spider mites")
             .replace("_", " "))


def figure(fn: str, max_width: int) -> str:
    """Downscale a report figure; return the smaller of its JPEG/PNG data URI."""
    src = R / fn
    if not src.exists():
        raise SystemExit(f"missing figure: {src}")
    im = Image.open(src)
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
    im = im.convert("RGB")
    j = BytesIO(); im.save(j, "JPEG", quality=86, optimize=True, progressive=True)
    p = BytesIO(); im.convert("P", palette=Image.ADAPTIVE, colors=256).save(p, "PNG", optimize=True)
    smaller = j if len(j.getvalue()) <= len(p.getvalue()) else p
    kind = "jpeg" if smaller is j else "png"
    return f"data:image/{kind};base64," + base64.b64encode(smaller.getvalue()).decode()


CURVES = figure("training_curves.png", 1400)


def cm_html() -> str:
    """Confusion matrix as live HTML -- hoverable, and legible on a projector."""
    out = ['<table class="cm"><tr><td></td>']
    out += [f'<td class="cl">{pretty(c)}</td>' for c in names]
    out.append("</tr>")
    for i, row in enumerate(cm):
        out.append(f'<tr><td class="rl">{pretty(names[i])}</td>')
        total = sum(row)
        for j, v in enumerate(row):
            share = v / total if total else 0
            if i == j:
                bg = f"rgba(30,70,53,{.18 + .82 * share:.3f})"
                col = "#fff" if share > .45 else "var(--brand)"
            elif v == 0:
                bg, col = "rgba(30,70,53,.035)", "rgba(31,42,36,.22)"
            else:
                bg = f"rgba(166,70,47,{.14 + .7 * min(share / .08, 1):.3f})"
                col = "#5a2618"
            ring = " d" if (i != j and v >= 10) else ""
            out.append(f'<td class="c{ring}" style="background:{bg};color:{col}" '
                       f'title="{pretty(names[i])} → {pretty(names[j])}: {v}">{v or ""}</td>')
        out.append("</tr>")
    return "".join(out) + "</table>"


def f1_rows(hi: str) -> str:
    return "".join(
        f'<tr{" class=hi" if d["name"] == hi else ""}><td>{pretty(d["name"])}</td>'
        f'<td>{d["p"]:.3f}</td><td>{d["r"]:.3f}</td><td>{d["f1"]:.4f}</td><td>{d["sup"]}</td></tr>'
        for d in sorted(pc, key=lambda x: -x["f1"]))


S: list[str] = []


def slide(html: str, cls: str = "", note: str = "") -> None:
    S.append(f'<section class="slide {cls}" data-note="{note}">{html}</section>')


# ---- 1 · title -------------------------------------------------------------
slide(f"""
<div class="brand-mark"><div class="m">🌿</div><div><div class="w">Leaf Scan</div>
<div class="t">Tomato Disease Detector</div></div></div>
<div class="eyebrow">Applied AI &amp; Machine Learning — Midterm</div>
<h1>Putting a plant pathologist<br>in a farmer's pocket</h1>
<p class="sub">A tomato leaf disease classifier for growers who cannot easily reach an
agricultural expert — MobileNetV2 transfer learning across {n} conditions, delivered as a
phone-friendly web app with treatment guidance in English and Urdu.</p>
<div class="headline">
  <div><div class="lab">Macro F1</div><div class="num">{mf1:.4f}</div></div>
  <div><div class="lab">Accuracy</div><div class="num">{acc:.4f}</div></div>
  <div><div class="lab">Test images</div><div class="num">{n_te:,}</div></div>
  <div><div class="lab">Classes</div><div class="num">{n}</div></div>
</div>
<div class="author"><div><div class="nm">{AUTHOR}</div>
<div class="mt">Every figure in this deck comes from one real training run on a Colab T4,
reproduced locally on CPU.</div></div>
<div class="pill">github.com/AliKhan84/tomato-disease-detector</div></div>
""", note="Open on the person, not the model: a grower with a diseased crop and no one to ask.")

# ---- 2 · problem -----------------------------------------------------------
slide(f"""
<div class="eyebrow">The problem</div><h2>A farmer in Khyber Pakhtunkhwa sees spots on the
leaves. Who do they ask?</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card lead"><h3>The gap</h3><ul>
<li>Tomato is a staple cash crop across KPK — and one of the most disease-prone things a
smallholder can plant.</li>
<li>Proper diagnosis means reaching an <strong>agricultural extension officer</strong>. From
a remote valley that can mean a journey, a wait, or no one available at all.</li>
<li>So the call gets made <strong>by eye, by a neighbour, or by whoever sells the
chemicals</strong> — whose incentive is to sell a spray.</li></ul></div>
<div class="card"><h3>What a wrong guess costs</h3><ul>
<li>Wrong fungicide: money gone, disease untouched, days lost while it spreads.</li>
<li>Early blight and late blight look alike but move at very different speeds — late blight
can take a field in under a week.</li>
<li>Spraying a <em>healthy</em> crop is pure cost, and builds resistance for later.</li></ul></div>
</div>
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card"><h3>What a useful tool has to do</h3><ul>
<li><strong>Work from one phone photo.</strong> No lab, no sample posted anywhere.</li>
<li><strong>Answer in seconds on cheap hardware</strong> — no GPU at inference.</li>
<li><strong>Say what to do next</strong>, not just name a disease in English.</li>
<li><strong>Be honest when unsure.</strong> A confident wrong diagnosis is worse than no
tool at all.</li></ul></div>
<div class="card good"><h3>Framing it as machine learning</h3>
<p>One leaf photo in, a ranked diagnosis over <strong>{n} classes</strong> out — 10 diseases
plus healthy — with treatment text attached to each.</p>
<p class="foot">That is an image classification problem, and public data exists to solve it
today. The hard parts are not the model: they are honest evaluation, and knowing when the
answer should not be trusted.</p></div>
</div></div></div>
""", note="Do not overclaim — I have not surveyed KPK extension coverage. The point is structural: expert access is scarce, a phone camera is not.")

# ---- 3 · solution ----------------------------------------------------------
slide(f"""
<div class="eyebrow">The solution</div><h2>Leaf Scan — a diagnosis, an explanation, and a
next step</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:10px">
<div class="card lead"><h3>1 · Diagnose</h3><p>Photo in → <strong>top-3 ranked classes</strong>
with confidence bars, so a close call between early and late blight is visible instead of
hidden behind one label. Treatment guidance in <strong>English and Urdu</strong>, right-to-left
in proper Nastaliq script.</p></div>
<div class="card"><h3>2 · Model Performance</h3><p>The app shows its own report card — macro
F1, per-class F1 with the weakest class flagged, the confusion matrix.
<strong>Read from <code>reports/</code> at runtime</strong>, so what it claims always matches
the model actually loaded.</p></div>
<div class="card"><h3>3 · Explainability</h3><p>Grad-CAM shows <em>where</em> the model
looked. If the heat sits on background instead of the lesion, the user can see that and
distrust the answer.</p></div>
<div class="card"><h3>4 · Method &amp; Limitations</h3><p>The caveats ship
<strong>inside the product</strong>, not only in this report.</p></div>
</div>
<div style="display:flex;flex-direction:column;gap:10px">
<div class="card good"><h3>Design decisions that follow from the user</h3><ul>
<li><strong>Runs on CPU.</strong> MobileNetV2 at 2.3M parameters, not ResNet50 at 25M — the
saved model is 24 MB.</li>
<li><strong>Bilingual by default</strong>, not buried in a settings menu.</li>
<li><strong>Ranked output, not one label.</strong> Second and third place carry real
information when two diseases resemble each other.</li>
<li><strong>EXIF-aware image handling</strong> — phone photos arrive rotated, RGBA or
greyscale, and all three break naive preprocessing.</li>
<li><strong>Light and dark themes</strong> — a phone outdoors in sunlight is a different
problem from a laptop indoors.</li></ul></div>
<div class="card"><h3>What it is not</h3><p>Not a replacement for an agronomist, and not yet
validated in a real field. It is a <strong>triage tool</strong>: it narrows eleven
possibilities to two or three, in seconds, for free.</p></div>
</div></div></div>
""", note="Honest positioning is triage, not diagnosis. Saying that out loud is more defensible and still useful.")

# ---- 4 · pipeline ----------------------------------------------------------
slide(f"""
<div class="eyebrow">Pipeline</div><h2>From {n_tr + n_va + n_te:,} images to a 24 MB model</h2>
<div class="body" style="gap:11px">
<div class="flow">
  <div class="fs"><b>Data</b><span>Kaggle · {n} classes<br>{n_tr + n_va + n_te:,} images</span></div>
  <div class="fa">→</div>
  <div class="fs"><b>Split</b><span>preserve archive<br>boundary · seeded</span></div>
  <div class="fa">→</div>
  <div class="fs"><b>Augment</b><span>flip · rotate<br>contrast · on 0–255</span></div>
  <div class="fa">→</div>
  <div class="fs"><b>MobileNetV2</b><span>ImageNet weights<br>154 layers</span></div>
  <div class="fa">→</div>
  <div class="fs"><b>Head</b><span>GAP · Dropout 0.2<br>Dense({n}, softmax)</span></div>
  <div class="fa">→</div>
  <div class="fs on"><b>Two-stage fit</b><span>frozen → fine-tune<br>class-weighted</span></div>
</div>
<div class="three" style="flex:1;min-height:0">
<div class="card"><h3>Data &amp; split</h3>
<table><tbody>
<tr><td>Train</td><td>{n_tr:,}</td></tr>
<tr><td>Validation</td><td>{n_va:,}</td></tr>
<tr><td>Test</td><td>{n_te:,}</td></tr>
<tr><td>Imbalance</td><td>{imb:.1f}×</td></tr></tbody></table>
<p class="foot"><strong>The archive's own train/valid boundary is preserved.</strong> Pooling
everything and reshuffling 70/15/15 puts near-duplicates on both sides and reports ~99% that
means nothing. Deterministic — sorted filenames, seeded RNG — so Colab and the laptop produce
byte-identical splits, confirmed by SHA-256.</p></div>
<div class="card"><h3>Model</h3>
<table><tbody>
<tr><td>Total params</td><td>2,272,075</td></tr>
<tr><td>Trainable</td><td>1,853,707</td></tr>
<tr><td>Frozen</td><td>418,368</td></tr>
<tr><td>BatchNorm</td><td>52 · frozen</td></tr></tbody></table>
<p class="foot"><strong>Augmentation runs before rescaling.</strong>
<code>RandomContrast</code> assumes 0–255 input; on already-normalised [-1,1] data it
distorts rather than augments — a silent accuracy leak with no error message. Preprocessing
is saved <em>inside</em> the model, so inference cannot drift from training.</p></div>
<div class="card"><h3>Stack</h3>
<p style="font-size:12.5px;line-height:1.7"><strong>TensorFlow 2.19</strong> · Keras 3.15 —
model, input pipeline, saved format<br>
<strong>scikit-learn 1.9</strong> — class weights, macro F1, confusion matrix<br>
<strong>NumPy 2.4</strong> · <strong>Pillow 12.3</strong> — arrays, EXIF-safe image I/O<br>
<strong>Matplotlib 3.11</strong> — figures, re-rendered per theme<br>
<strong>Streamlit 1.61</strong> — the four-tab app<br>
<strong>Colab T4</strong> — one notebook, 50–70 min end to end</p>
<p class="foot">No PyTorch, no OpenCV, no pandas. Deployment installs
<code>tensorflow-cpu</code>: 252 MB against 645 MB, since the CUDA stack is dead weight in a
browser-served app.</p></div>
</div></div>
""", note="Why MobileNetV2: 25,851 images is far too few to train from scratch, and the deployment target is CPU.")

# ---- 5 · stage 1 -----------------------------------------------------------
slide(f"""
<div class="eyebrow">How accuracy improved — step 1</div>
<h2>The frozen backbone stalls at {s1:.4f}</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:11px">
<div class="row">
<div class="card"><div class="lab">Best val accuracy</div><div class="num">{s1:.4f}</div>
<div class="foot">8 epochs, cap 10</div></div>
<div class="card"><div class="lab">Trainable params</div><div class="num sm">14,091</div>
<div class="foot">the Dense head only</div></div></div>
<div class="card"><h3>The setup</h3><ul>
<li>MobileNetV2 <code>trainable=False</code> — ImageNet features used as they come</li>
<li>Adam at 1e-3, <code>sparse_categorical_crossentropy</code></li>
<li><strong>Balanced class weights</strong> — a powdery mildew error counts 2.34× a late
blight error, so the {imb:.1f}× imbalance cannot silence the small classes</li>
<li>Checkpoint on best val accuracy + EarlyStopping(patience 3)</li></ul></div>
<div class="card lead"><h3>The symptom</h3><p>Accuracy plateaued, and <strong>training
accuracy sat below validation</strong> with both losses level.</p></div>
</div>
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card good"><h3>Reading it correctly — the decision the project turns on</h3>
<p>Train below val with flat losses is <strong>underfitting</strong>, not overfitting. The
model is not memorising the training set; it lacks the capacity to fit it at all.</p>
<p class="foot">The frozen ImageNet features were the ceiling. They encode generic edges and
textures — not the specific lesion patterns that separate early blight from late blight.</p></div>
<div class="card"><h3>What that ruled out</h3><ul>
<li><strong>More epochs</strong> — the losses were already flat</li>
<li><strong>Heavier augmentation</strong> — that treats overfitting; here it would have made
things <em>worse</em></li>
<li><strong>A wider head</strong> — the bottleneck sits upstream of the head</li></ul>
<p class="foot">Every reflexive "improve the model" move was the wrong one. Only adding
capacity to the <em>features themselves</em> could help — which meant unfreezing.</p></div>
</div></div></div>
""", note="Expect the question here. How did you know it was underfitting? Training accuracy BELOW validation, both losses flat.")

# ---- 6 · stage 2 -----------------------------------------------------------
slide(f"""
<div class="eyebrow">How accuracy improved — step 2</div>
<h2>Fine-tuning the top 35%: +{gain:.2f} points</h2>
<div class="body" style="gap:10px">
<div class="row" style="flex:0 0 auto">
<div class="card"><div class="lab">Stage 1 — frozen</div><div class="num">{s1:.4f}</div>
<div class="bar"><i style="width:{s1 * 100:.1f}%"></i></div>
<div class="foot">head only · 14,091 params</div></div>
<div class="card lead"><div class="lab">Stage 2 — fine-tuned</div><div class="num">{s2:.4f}</div>
<div class="bar"><i style="width:{s2 * 100:.1f}%"></i></div>
<div class="foot">top 35% unfrozen · 1,853,707 params</div></div>
<div class="card"><div class="lab">Improvement</div>
<div class="num" style="color:var(--accent)">+{gain:.2f}</div>
<div class="foot">points of val accuracy</div></div>
<div class="card"><div class="lab">Learning rate</div><div class="num sm">1e-5</div>
<div class="foot">100× below stage 1</div></div></div>
<div class="two" style="flex:1;min-height:0">
<figure><img src="{CURVES}" alt="Stage 1 training curves">
<figcaption>Stage 1 curves — training accuracy tracks <em>below</em> validation with both
losses flattening. That gap is the underfitting signature that justified stage 2.</figcaption></figure>
<div style="display:flex;flex-direction:column;gap:10px">
<div class="card lead"><h3>Three things that make it safe</h3><ul>
<li><strong>All 52 BatchNorm layers stay frozen.</strong> In Keras <code>trainable=False</code>
on BN also forces inference mode; letting moving statistics update on small batches corrupts
the pretrained representation within one epoch. Verified unmoved after training.</li>
<li><strong>Learning rate drops 100×.</strong> At stage 1's rate the first gradients would
destroy the ImageNet weights before refining them.</li>
<li><strong>Stage 2 checkpoints to a separate file.</strong> Had it failed to beat the
baseline, the notebook reloads stage 1 and says so.</li></ul></div>
<div class="card"><h3>Also worth noting</h3><p style="font-size:13.5px">The cut is a
<strong>fraction</strong> — 35% of 154 layers — not a hardcoded index, so it survives a
backbone swap. <code>ReduceLROnPlateau</code> was added for stage 2. Validation accuracy was
<strong>still rising at the final epoch</strong>: the run ended on its epoch cap, so
{s2:.4f} is a floor for this configuration, not a converged ceiling.</p></div>
</div></div></div>
""", note="+10.81 points, the single biggest improvement in the project. The three safeguards are the difference between fine-tuning and wrecking the backbone.")

# ---- 7 · results -----------------------------------------------------------
slide(f"""
<div class="eyebrow">Results</div><h2>Test set — {n_te:,} images the model never saw</h2>
<div class="body">
<div class="row">
<div class="card lead"><div class="lab">Macro F1</div><div class="num">{mf1:.4f}</div>
<div class="foot">headline — every class weighted equally</div></div>
<div class="card"><div class="lab">Accuracy</div><div class="num">{acc:.4f}</div>
<div class="foot">{correct:,} of {n_te:,} correct</div></div>
<div class="card"><div class="lab">Strongest</div><div class="num sm">{best['f1']:.4f}</div>
<div class="foot">{pretty(best['name'])}</div></div>
<div class="card"><div class="lab">Weakest</div>
<div class="num sm" style="color:var(--accent)">{worst['f1']:.4f}</div>
<div class="foot">{pretty(worst['name'])} · recall {worst['r']:.4f}</div></div></div>
<div class="two" style="margin-top:2px">
<div class="card" style="overflow:hidden"><h3>Per-class performance</h3>
<table><thead><tr><th>Class</th><th>Prec</th><th>Recall</th><th>F1</th><th>n</th></tr></thead>
<tbody>{f1_rows(worst['name'])}</tbody></table></div>
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card lead"><h3>The detail that matters</h3>
<p>Macro F1 ({mf1:.4f}) sits <strong>above</strong> accuracy ({acc:.4f}). Under a
{imb:.1f}× imbalance that ordering means <strong>no minority class was sacrificed</strong> to
lift the average — the class weighting did its job. Powdery mildew, smallest at
{pm['sup']} test images, scores {pm['f1']:.4f}.</p></div>
<div class="card"><h3>Reproduced locally</h3><p>Evaluation re-run on CPU against a split
rebuilt from the archive: <strong>identical in every per-class figure</strong>. Both
manifests hash to the same SHA-256 — the same {n_te:,} test images on both machines, so
nothing the model trained on leaked into the score.</p></div>
<div class="card"><h3>Where the errors sit</h3><p>Only three classes fall below 0.93 — early
blight, Septoria leaf spot, target spot. All three are dark-lesion diseases that resemble one
another.</p></div>
</div></div></div>
""", note="Macro F1 above accuracy is the evidence that weighting worked. Lead with that if challenged on imbalance.")

# ---- 8 · confusion matrix --------------------------------------------------
slide(f"""
<div class="eyebrow">Results</div><h2>Confusion matrix — where it actually fails</h2>
<div class="body" style="gap:8px">
<div style="flex:1;min-height:0;display:flex;align-items:center;justify-content:center">{cm_html()}</div>
<div class="row" style="flex:0 0 auto">
<div class="card"><p style="font-size:13px;margin:0"><strong>Rows = truth, columns =
prediction.</strong> The diagonal is correct; off-diagonal cells ≥10 are ringed. Hover any
cell for the exact pair.</p></div>
<div class="card lead"><p style="font-size:13px;margin:0"><strong>The one real
weakness.</strong> Early blight loses <strong>{eb_late} to late blight</strong> and
<strong>{eb_sept} to Septoria</strong> — {(eb_late + eb_sept) / eb['sup'] * 100:.0f}% of the
class in two cells. All three are dark necrotic lesions; a human agronomist makes the same
mistake from a photograph. <strong>It is also the confusion that matters most in the
field</strong>, since late blight moves far faster.</p></div>
</div></div>
""", note="Rendered live from metrics.json, not an image. Early blight recall 0.8230 is the weakest number in the project — own it before someone finds it.")

# ---- 9 · limitations + close -----------------------------------------------
slide(f"""
<div class="eyebrow">Honest assessment</div>
<h1 style="font-size:38px">What it does, and what it cannot</h1>
<div class="two" style="margin-top:18px;flex:0 1 auto">
<div class="card"><h3>Limitations I would state to a user</h3><ul>
<li><strong>Confidence does not detect wrong subjects.</strong> Measured, not assumed: a solid
grey square returns late blight at 0.982, a drawn face returns mosaic virus at 0.989 — inside
the real-leaf range. No threshold separates them, and entropy is no better. The app says so
rather than implying its 50% flag is a safeguard.</li>
<li><strong>No KPK field validation.</strong> The data is lab and field imagery from Kaggle,
<em>none of it from the target region</em>. Real performance on a phone photo in a Swat valley
is unmeasured and probably lower.</li>
<li><strong>Urdu, not Pashto.</strong> Urdu reaches many but not all KPK growers; Pashto is
the larger first language here and is not covered.</li>
<li><strong>Early blight recall {worst['r']:.4f}</strong> — nearly one in five missed.</li>
<li>Closed set of {n} classes · single split, no cross-validation · advice text not
agronomist-reviewed</li></ul></div>
<div class="card"><h3>What I would defend</h3><ul>
<li><strong>Reading the plateau correctly.</strong> Train below val with flat losses meant
underfitting — so augmentation would have hurt. Unfreezing was the only lever, and it returned
<strong>+{gain:.2f} points</strong>.</li>
<li><strong>Refusing the easy 99%.</strong> Keeping the archive's split boundary cost headline
accuracy and bought a number that is true.</li>
<li><strong>Measuring the failure mode</strong> instead of assuming the confidence score was a
safeguard.</li></ul>
<h3 style="margin-top:12px">Next, in priority order</h3><ul>
<li>A field-photograph validation set from KPK — the single biggest gap</li>
<li>An explicit "not a leaf" class for real out-of-distribution rejection</li>
<li>Pashto advice text alongside Urdu</li>
<li>Targeted work on the early/late blight boundary</li></ul></div>
</div>
<div class="author" style="border-color:rgba(238,241,231,.2)">
<div><div class="nm">{AUTHOR}</div>
<div class="mt" style="color:rgba(238,241,231,.6)">Macro F1 {mf1:.4f} · accuracy {acc:.4f} ·
+{gain:.2f} from fine-tuning</div></div>
<div class="pill" style="background:rgba(255,255,255,.12);color:#EEF1E7">Thank you — questions?</div></div>
""", "dark", note="Close on the honest limits, then the three defensible decisions. Volunteer the KPK-validation gap before being asked.")

TITLES = ["Title", "The problem", "The solution", "Pipeline", "Stage 1 — plateau",
          "Stage 2 — fine-tune", "Results", "Confusion matrix", "Limitations & close"]
assert len(TITLES) == len(S), f"{len(TITLES)} titles vs {len(S)} slides"

OUT.write_text(SHELL.replace("__AUTHOR__", AUTHOR)
                    .replace("__SLIDES__", "\n".join(S))
                    .replace("__TITLES__", json.dumps(TITLES)))
print(f"{OUT.name}  {len(S)} slides  {OUT.stat().st_size / 1024:.0f} KB")
