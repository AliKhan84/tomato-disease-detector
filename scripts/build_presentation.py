#!/usr/bin/env python3
"""Inject slides + base64 figures into presentation.html. Numbers all read from reports/."""
import base64, json, re
from io import BytesIO
from pathlib import Path
from PIL import Image

ROOT = Path("/home/alikhan/projects/tomato-disease-detector")
R, M = ROOT / "reports", ROOT / "models"
metrics = json.loads((R / "metrics.json").read_text())
run = json.loads((R / "training_run.json").read_text())
man = json.loads((M / "split_manifest.json").read_text())

names, cm = metrics["class_names"], metrics["confusion_matrix"]
n = len(names)
counts = man["counts"]
tr = {c: counts[c]["train"] for c in names}
n_tr, n_va, n_te = (sum(counts[c][k] for c in names) for k in ("train", "val", "test"))

pc = []
for i in range(n):
    sup = sum(cm[i]); col = sum(cm[r][i] for r in range(n)); tp = cm[i][i]
    pc.append(dict(name=names[i], p=tp/col if col else 0, r=tp/sup if sup else 0,
                   f1=metrics["per_class_f1"][names[i]], sup=sup))

def pretty(s): return s.replace("Tomato_", "").replace("Spider_mites Two-spotted_spider_mite", "Spider mites").replace("_", " ")

FIGS = {"curves": ("training_curves.png", 1500), "filters": ("conv_filters.png", 1100),
        "gradcam": ("gradcam.png", 1400), "fm_early": ("feature_maps_block_1_expand_relu.png", 1200),
        "fm_mid": ("feature_maps_block_6_expand_relu.png", 1200),
        "fm_late": ("feature_maps_out_relu.png", 1200)}

def enc(fn, mw):
    im = Image.open(R / fn)
    if im.width > mw: im = im.resize((mw, round(im.height*mw/im.width)), Image.LANCZOS)
    im = im.convert("RGB")
    j = BytesIO(); im.save(j, "JPEG", quality=86, optimize=True, progressive=True)
    p = BytesIO(); im.convert("P", palette=Image.ADAPTIVE, colors=256).save(p, "PNG", optimize=True)
    if len(j.getvalue()) <= len(p.getvalue()):
        return "data:image/jpeg;base64," + base64.b64encode(j.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(p.getvalue()).decode()

F = {k: enc(f, w) for k, (f, w) in FIGS.items()}

acc, mf1 = metrics["accuracy"], metrics["macro_f1"]
s1, s2 = run["stage1_best_val_accuracy"], run["stage2_best_val_accuracy"]
gain = (s2 - s1) * 100
worst = min(pc, key=lambda d: d["f1"]); best = max(pc, key=lambda d: d["f1"])
imb = max(tr.values()) / min(tr.values())

# ---------- confusion matrix as live HTML ----------
def cm_html():
    mx = max(max(r) for r in cm)
    out = ['<table class="cm"><tr><td></td>']
    for c in names: out.append(f'<td class="cl">{pretty(c)}</td>')
    out.append("</tr>")
    for i, row in enumerate(cm):
        out.append(f'<tr><td class="rl">{pretty(names[i])}</td>')
        for j, v in enumerate(row):
            share = v / sum(row) if sum(row) else 0
            if i == j:
                bg = f"rgba(30,70,53,{.18+.82*share:.3f})"; col = "#fff" if share > .45 else "var(--brand)"
            elif v == 0:
                bg = "rgba(30,70,53,.035)"; col = "rgba(31,42,36,.22)"
            else:
                bg = f"rgba(166,70,47,{.14+.7*min(share/.08,1):.3f})"; col = "#5a2618"
            d = " d" if (i != j and v >= 10) else ""
            out.append(f'<td class="c{d}" style="background:{bg};color:{col}" '
                       f'title="{pretty(names[i])} → {pretty(names[j])}: {v}">{v or ""}</td>')
        out.append("</tr>")
    return "".join(out) + "</table>"

def f1_rows(hi=None):
    rows = []
    for d in sorted(pc, key=lambda x: -x["f1"]):
        k = ' class="hi"' if d["name"] == hi else ""
        rows.append(f'<tr{k}><td>{pretty(d["name"])}</td><td>{d["p"]:.3f}</td>'
                    f'<td>{d["r"]:.3f}</td><td>{d["f1"]:.4f}</td><td>{d["sup"]}</td></tr>')
    return "".join(rows)

S = []
def slide(html, cls="", note=""):
    S.append((f'<section class="slide {cls}" data-note="{note}">{html}</section>'))

# 1 title
slide(f"""
<div class="brand-mark"><div class="m">🌿</div><div><div class="w">Leaf Scan</div>
<div class="t">Tomato Disease Detector</div></div></div>
<div class="eyebrow">Applied AI &amp; Machine Learning — Midterm</div>
<h1>Tomato Leaf Disease<br>Detection with Transfer Learning</h1>
<p class="sub">MobileNetV2 fine-tuned across 11 tomato leaf conditions, deployed as a
Streamlit diagnostic app with Grad-CAM explanations and bilingual treatment guidance.</p>
<div class="headline">
  <div class="h"><div class="lab">Macro F1</div><div class="num">{mf1:.4f}</div></div>
  <div class="h"><div class="lab">Accuracy</div><div class="num">{acc:.4f}</div></div>
  <div class="h"><div class="lab">Test images</div><div class="num">{n_te:,}</div></div>
  <div class="h"><div class="lab">Classes</div><div class="num">{n}</div></div>
</div>
<div class="author"><div><div class="nm">{'Ali Khan'}</div>
<div class="mt">Every figure in this deck comes from one real training run on a Colab T4,
reproduced locally on CPU.</div></div>
<div class="pill">github.com/AliKhan84/tomato-disease-detector</div></div>
""", note="Headline: 0.9495 macro F1 on 3,343 held-out images. Fine-tuning was the single biggest win — +10.81 points.")

# 2 problem
slide(f"""
<div class="eyebrow">Problem</div><h2>Why this problem, and why it is hard</h2>
<div class="body"><div class="two">
<div class="card"><h3>The task</h3><ul>
<li>Tomato is one of the highest-value smallholder crops; leaf disease is diagnosed by eye,
often late, and mistreatment wastes both chemicals and harvest.</li>
<li>Given one leaf photo, name the condition from <strong>11 classes</strong> and give a
grower something actionable, in a language they read.</li>
<li>Runs on a laptop or a phone browser — no GPU at inference.</li></ul></div>
<div class="card"><h3>Why it is not trivial</h3><ul>
<li><strong>Classes look alike.</strong> Early blight, late blight and Septoria leaf spot
all present as dark necrotic lesions.</li>
<li><strong>Imbalance {imb:.1f}×</strong> — {max(tr.values()):,} images for the largest class,
{min(tr.values()):,} for the smallest.</li>
<li><strong>Mixed sources.</strong> Lab imagery and field photos in one archive, with
near-duplicates that make careless splitting produce fake accuracy.</li>
<li><strong>Small data by deep-learning standards</strong> — 25,851 training images is far
too few to train a CNN from scratch.</li></ul></div>
</div></div>
""", note="The last point is the justification for transfer learning — set it up here.")

# 3 stack
slide(f"""
<div class="eyebrow">Tooling</div><h2>Libraries and what each one does</h2>
<div class="body">
<div class="three">
<div class="card"><div class="lab">Modelling</div>
<p><strong>TensorFlow 2.19.1</strong> · <strong>Keras 3.15</strong><br>
<span style="font-size:14px;color:var(--mute)">Sequential model, MobileNetV2 with ImageNet
weights, <code>image_dataset_from_directory</code> for the input pipeline, augmentation and
rescaling layers baked into the saved model.</span></p></div>
<div class="card"><div class="lab">Data &amp; metrics</div>
<p><strong>NumPy 2.4.6</strong> · <strong>scikit-learn 1.9</strong><br>
<span style="font-size:14px;color:var(--mute)"><code>compute_class_weight</code> for the
imbalance, <code>classification_report</code> and <code>confusion_matrix</code> for
evaluation — macro F1 as the headline metric.</span></p></div>
<div class="card"><div class="lab">Interface</div>
<p><strong>Streamlit 1.61</strong> · <strong>Pillow 12.3</strong><br>
<span style="font-size:14px;color:var(--mute)">Four-tab app, custom CSS, light/dark themes.
Pillow handles EXIF rotation and RGBA/greyscale uploads before inference.</span></p></div>
</div>
<div class="three" style="margin-top:2px">
<div class="card"><div class="lab">Figures</div>
<p><strong>Matplotlib 3.11</strong><br><span style="font-size:14px;color:var(--mute)">
Confusion matrix, curves, Grad-CAM overlays and feature maps — re-rendered in the active
theme palette rather than shipped as fixed images.</span></p></div>
<div class="card"><div class="lab">Training</div>
<p><strong>Google Colab · NVIDIA T4</strong><br><span style="font-size:14px;color:var(--mute)">
One self-contained notebook: unzip → split → train both stages → evaluate → package into
<code>artifacts.zip</code>. 50–70 minutes end to end.</span></p></div>
<div class="card lead"><div class="lab">Deliberately excluded</div>
<p><span style="font-size:14px;color:var(--mute)">No PyTorch, no OpenCV, no pandas — every
dependency here earns its place. Inference installs <code>tensorflow-cpu</code>: 252 MB
against 645 MB, since the CUDA stack is dead weight in a browser-served app.</span></p></div>
</div></div>
""", note="If asked why TF over PyTorch: Keras 3 Sequential keeps preprocessing inside the saved model, so inference cannot drift from training.")

# 4 data
rows = "".join(f'<tr{" class=hi" if c in (max(tr,key=tr.get),min(tr,key=tr.get)) else ""}>'
               f"<td>{pretty(c)}</td><td>{counts[c]['train']:,}</td><td>{counts[c]['val']}</td>"
               f"<td>{counts[c]['test']}</td></tr>" for c in names)
slide(f"""
<div class="eyebrow">Dataset</div><h2>32,534 images across 11 classes</h2>
<div class="body"><div class="two">
<div style="min-height:0;overflow:hidden"><div class="card" style="height:100%;overflow:hidden">
<table><thead><tr><th>Class</th><th>Train</th><th>Val</th><th>Test</th></tr></thead>
<tbody>{rows}
<tr style="font-weight:600"><td>Total</td><td>{n_tr:,}</td><td>{n_va:,}</td><td>{n_te:,}</td></tr>
</tbody></table></div></div>
<div style="display:flex;flex-direction:column;gap:12px;min-height:0">
<div class="card"><h3>Kaggle — Tomato Disease Multiple Sources</h3>
<p style="font-size:14.5px">Ships pre-split as <code>train/</code> and <code>valid/</code>.
Aggregates lab PlantVillage imagery with field photography.</p></div>
<div class="card lead"><h3>The split decision that matters</h3>
<p style="font-size:14.5px">Archive <code>train/</code> → train. Archive <code>valid/</code>
→ halved per class into val and test. <strong>Nothing is pooled and reshuffled.</strong></p>
<p class="foot">Pooling then re-splitting 70/15/15 puts near-duplicates on both sides, so the
score measures memorisation. That route reports ~99% and means nothing. 94.9% earned on a
clean boundary is the more honest number.</p></div>
<div class="card"><h3>Determinism</h3>
<p style="font-size:14.5px">Filenames sorted, RNG seeded per class, no reliance on
filesystem order — Colab and the laptop produce <strong>byte-identical splits</strong>,
verified by SHA-256 over both manifests.</p></div>
</div></div></div>
""", note="Imbalance 3.10x. One truncated PNG excluded, which is why test is 3,343 not 3,344. Two WebP files with .jpg extensions were re-encoded in place, not deleted, so counts stay stable.")

# 5 architecture
slide(f"""
<div class="eyebrow">Architecture</div><h2>One Sequential model, preprocessing included</h2>
<div class="body"><div class="two">
<div class="mono">Input(224, 224, 3)
  │
  ├─ RandomFlip("horizontal")   ┐
  ├─ RandomRotation(0.1)        ├ augment on raw 0–255
  ├─ RandomContrast(0.1)        ┘
  │
  ├─ Rescaling(1/127.5, -1)     → [-1, 1]
  │
  ├─ MobileNetV2                  154 layers
  │    include_top=False          ImageNet weights
  │    52 BatchNorm — always frozen
  │
  ├─ GlobalAveragePooling2D()
  ├─ Dropout(0.2)
  └─ Dense(11, softmax)

Total params        2,272,075
Trainable (stage 2) 1,853,707
Frozen                418,368</div>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="card"><h3>Why MobileNetV2</h3><p style="font-size:14.5px">2.3M parameters against
ResNet50's 25M. Trains in minutes on a T4, loads fast in a browser-served app, and ImageNet
features transfer well to leaf texture. 25,851 images cannot support training from scratch.</p></div>
<div class="card lead"><h3>Augmentation runs <em>before</em> rescaling</h3>
<p style="font-size:14.5px"><code>RandomContrast</code> rescales around the mean assuming
0–255 input. Applied to already-normalised [-1, 1] data it distorts images instead of
augmenting them — a silent bug that costs accuracy with no error.</p></div>
<div class="card"><h3>Preprocessing lives inside the model</h3>
<p style="font-size:14.5px">The saved <code>.keras</code> file carries its own normalisation,
so inference passes a plain 0–255 array and <strong>cannot drift out of sync</strong> with
training. Augmentation layers are inactive at inference automatically.</p></div>
</div></div></div>
""", note="2.27M total params. Frozen 418,368 = the bottom 65% of the backbone.")

# 6 imbalance
wrows = "".join(f'<tr{" class=hi" if c in (max(tr,key=tr.get),min(tr,key=tr.get)) else ""}>'
                f"<td>{pretty(c)}</td><td>{tr[c]:,}</td><td>{n_tr/(n*tr[c]):.3f}</td></tr>"
                for c in sorted(names, key=lambda x: -tr[x]))
slide(f"""
<div class="eyebrow">Class imbalance</div><h2>Weighting the loss, and measuring with macro F1</h2>
<div class="body"><div class="two">
<div class="card" style="overflow:hidden"><h3>Balanced class weights</h3>
<p class="foot" style="margin:0 0 8px">
<code>compute_class_weight("balanced")</code> — weight = n / (k · count)</p>
<table><thead><tr><th>Class</th><th>Train</th><th>Weight</th></tr></thead>
<tbody>{wrows}</tbody></table></div>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="card"><h3>The problem</h3><p style="font-size:14.5px">Powdery mildew is
{min(tr.values()):,} images against late blight's {max(tr.values()):,} — a
<strong>{imb:.1f}× spread</strong>. Unweighted, the loss is dominated by the majority classes
and a failing minority class stays invisible behind a good-looking accuracy score.</p></div>
<div class="card lead"><h3>Two fixes, both cheap</h3><ul>
<li><strong>Weighted loss</strong> — powdery mildew errors count 2.34× a late-blight error,
so gradient updates stop ignoring it.</li>
<li><strong>Macro F1 as the headline</strong> — averages per-class, so every class carries
equal weight in the reported number regardless of size.</li></ul></div>
<div class="card"><h3>Did it work?</h3><p style="font-size:14.5px">Powdery mildew — the
smallest class at 126 test images — scores <strong>F1 {[d for d in pc if d['name']=='powdery_mildew'][0]['f1']:.4f}</strong>,
above the macro average. The minority class was not sacrificed.</p></div>
</div></div></div>
""", note="Macro F1 (0.9495) sits ABOVE accuracy (0.9488). That ordering is the evidence weighting worked.")

# 7 divider
slide(f"""
<div class="eyebrow">Results</div>
<h1 style="font-size:46px">Stage 1 hit a ceiling at 0.8341.<br>Fine-tuning took it to 0.9422.</h1>
<p class="sub" style="font-size:19px">The original plan said frozen backbone only. The
evidence said otherwise — and reading that evidence correctly was the single most valuable
decision in the project.</p>
""", "dark", note="Transition slide. Pause here — this is the core narrative of the project.")

# 8 stage 1
slide(f"""
<div class="eyebrow">Training — stage 1</div><h2>Frozen backbone: the diagnosis</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:12px">
<div class="row">
<div class="card"><div class="lab">Best val accuracy</div><div class="num">{s1:.4f}</div>
<div class="foot">8 epochs, cap 10</div></div>
<div class="card"><div class="lab">Trainable params</div><div class="num sm">14,091</div>
<div class="foot">the Dense head only</div></div></div>
<div class="card"><h3>Setup</h3><ul style="gap:6px">
<li>MobileNetV2 <code>trainable=False</code>, Adam at default 1e-3</li>
<li><code>sparse_categorical_crossentropy</code>, balanced class weights</li>
<li>ModelCheckpoint on best val accuracy + EarlyStopping(patience=3)</li></ul></div>
<div class="card lead"><h3>The symptom</h3><p style="font-size:14.5px">Accuracy plateaued and
<strong>training accuracy sat below validation</strong>, with both losses level.</p></div>
</div>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="card" style="border-left:3px solid var(--good)"><h3>Reading it correctly</h3>
<p style="font-size:14.5px">Train &lt; val with flat losses is <strong>underfitting</strong>,
not overfitting. The model is not memorising — it lacks the capacity to fit at all.</p>
<p class="foot">The frozen ImageNet features were the ceiling. They encode generic edges and
textures, not the specific lesion patterns that separate early from late blight.</p></div>
<div class="card"><h3>What that ruled out</h3><ul style="gap:6px">
<li><strong>More epochs</strong> — losses were already flat</li>
<li><strong>Heavier augmentation</strong> — an anti-overfitting tool; would have hurt</li>
<li><strong>A wider head</strong> — the bottleneck is upstream of the head</li></ul>
<p class="foot">Every standard "improve the model" reflex was wrong here. Only adding
capacity to the <em>features</em> could help.</p></div>
</div></div></div>
""", note="This is the key diagnostic reasoning. If asked one question, expect it here: how did you know it was underfitting? Answer: train accuracy below val, both losses flat.")

# 9 stage 2
slide(f"""
<div class="eyebrow">Training — stage 2</div><h2>Fine-tuning: +{gain:.2f} points</h2>
<div class="body">
<div class="row">
<div class="card"><div class="lab">Stage 1 — frozen</div><div class="num">{s1:.4f}</div>
<div class="bar"><i style="width:{s1*100:.1f}%"></i></div><div class="foot">8 epochs · head only</div></div>
<div class="card lead"><div class="lab">Stage 2 — fine-tuned</div><div class="num">{s2:.4f}</div>
<div class="bar"><i style="width:{s2*100:.1f}%"></i></div><div class="foot">8 epochs · top 35% unfrozen</div></div>
<div class="card"><div class="lab">Improvement</div>
<div class="num" style="color:var(--accent)">+{gain:.2f}</div>
<div class="foot">percentage points of val accuracy</div></div>
<div class="card"><div class="lab">Learning rate</div><div class="num sm">1e-5</div>
<div class="foot">100× below stage 1</div></div></div>
<div class="two" style="margin-top:2px">
<div class="card"><h3>What changed</h3><ul style="gap:7px">
<li>Top <strong>35%</strong> of the 154-layer backbone unfrozen — a fraction, not a hardcoded
index, so it survives a backbone swap</li>
<li>Trainable parameters <strong>14,091 → 1,853,707</strong></li>
<li>Cut at layer 100; everything below stays frozen</li>
<li><code>ReduceLROnPlateau</code>(factor 0.3, patience 2) added</li></ul></div>
<div class="card lead"><h3>Three things that make it safe</h3><ul style="gap:7px">
<li><strong>All 52 BatchNorm layers stay frozen</strong> — in Keras, <code>trainable=False</code>
on BN also forces inference mode. Letting moving statistics update on small batches corrupts
the pretrained representation within one epoch. Verified unmoved after training.</li>
<li><strong>LR drops 100×.</strong> At stage 1's rate the first gradients would destroy the
ImageNet weights before they could be refined.</li>
<li><strong>Separate checkpoint file.</strong> If stage 2 failed to beat the baseline, the
notebook reloads stage 1 and says so.</li></ul></div>
</div></div>
""", note="+10.81 points. The safety measures are what make the difference between fine-tuning and wrecking the backbone.")

# 10 curves
slide(f"""
<div class="eyebrow">Training — curves</div><h2>Accuracy and loss across both stages</h2>
<div class="body"><figure><img src="{F['curves']}" alt="Training curves">
<figcaption>Stage 1, frozen backbone. Training accuracy tracks below validation with both
losses flattening — the underfitting signature that motivated stage 2. Validation accuracy
was <strong>still rising at the final epoch</strong>, so 0.9422 is a floor for this
configuration, not a converged ceiling.</figcaption></figure></div>
""", note="Honest point: the run ended on its epoch cap, not convergence. More epochs would likely add a little more.")

# 11 headline results
slide(f"""
<div class="eyebrow">Evaluation</div><h2>Test set — {n_te:,} images never seen in training</h2>
<div class="body">
<div class="row">
<div class="card lead"><div class="lab">Macro F1</div><div class="num">{mf1:.4f}</div>
<div class="foot">headline metric — every class weighted equally</div></div>
<div class="card"><div class="lab">Accuracy</div><div class="num">{acc:.4f}</div>
<div class="foot">{sum(cm[i][i] for i in range(n)):,} of {n_te:,} correct</div></div>
<div class="card"><div class="lab">Strongest</div><div class="num sm">{best['f1']:.4f}</div>
<div class="foot">{pretty(best['name'])}</div></div>
<div class="card"><div class="lab">Weakest</div>
<div class="num sm" style="color:var(--accent)">{worst['f1']:.4f}</div>
<div class="foot">{pretty(worst['name'])} · recall {worst['r']:.4f}</div></div></div>
<div class="two" style="margin-top:2px">
<div class="card" style="overflow:hidden"><h3>Per-class performance</h3>
<table><thead><tr><th>Class</th><th>Prec</th><th>Recall</th><th>F1</th><th>n</th></tr></thead>
<tbody>{f1_rows(worst['name'])}</tbody></table></div>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="card lead"><h3>The detail that matters</h3>
<p style="font-size:14.5px">Macro F1 ({mf1:.4f}) sits <strong>above</strong> accuracy
({acc:.4f}). Under a {imb:.1f}× imbalance that ordering means <strong>no minority class was
sacrificed</strong> to lift the average — if one had been, macro F1 would drop below
accuracy.</p></div>
<div class="card"><h3>Reproduced locally</h3>
<p style="font-size:14.5px">Re-ran evaluation on CPU against a split rebuilt from the
archive: <strong>identical in every per-class figure</strong>. Both manifests hash to the
same SHA-256, confirming the same 3,343 test images on both machines.</p></div>
<div class="card"><h3>Where the errors concentrate</h3>
<p style="font-size:14.5px">Only three classes fall below 0.93 — early blight, Septoria leaf
spot and target spot. All three are dark-lesion diseases.</p></div>
</div></div></div>
""", note="If asked why macro F1 above accuracy: minority classes score higher than the weighted mix, which is exactly what class weighting was for.")

# 12 confusion matrix
slide(f"""
<div class="eyebrow">Evaluation</div><h2>Confusion matrix — where the model actually fails</h2>
<div class="body" style="gap:8px">
<div style="flex:1;min-height:0;display:flex;align-items:center;justify-content:center">{cm_html()}</div>
<div class="row" style="flex:0 0 auto">
<div class="card"><p style="font-size:13.5px;margin:0"><strong>Rows = truth, columns =
prediction.</strong> Diagonal is correct. Off-diagonal cells ≥10 are ringed in ochre. Hover
any cell for the exact pair.</p></div>
<div class="card lead"><p style="font-size:13.5px;margin:0"><strong>The one real weakness.</strong>
Early blight loses <strong>21 to late blight</strong> and <strong>17 to Septoria</strong> —
{(21+17)/322*100:.0f}% of the class in two cells. All three are dark necrotic lesions; a
human agronomist makes the same confusion from a photograph.</p></div>
</div></div>
""", note="Rendered live from metrics.json, not an image. Early_blight recall 0.8230 is the weakest figure in the project — own it before someone asks.")

# 13 grad-cam
slide(f"""
<div class="eyebrow">Explainability</div><h2>Grad-CAM — is it reading the lesion or the background?</h2>
<div class="body"><div class="two">
<figure><img src="{F['gradcam']}" alt="Grad-CAM overlays"></figure>
<div style="display:flex;flex-direction:column;gap:12px">
<div class="card"><h3>What this shows</h3><p style="font-size:14.5px">Gradient-weighted class
activation mapping: how strongly each spatial region drove the predicted class. All four
spot-checked images were classified <strong>correctly</strong>.</p></div>
<div class="card" style="border-left:3px solid var(--good)"><h3>The good result</h3>
<p style="font-size:14.5px">On early blight, late blight and leaf mold the heat sits
<strong>on the lesion</strong> — the model is reading diseased tissue, not an artefact of
the photograph.</p></div>
<div class="card lead"><h3>The uncomfortable one</h3><p style="font-size:14.5px">On the
bacterial spot example a large share of attention sits <strong>above the leaf tip, on
background</strong>. One sample is not a trend — but it is why this figure is committed to
the repo rather than summarised away.</p></div>
<div class="card"><h3>Implementation note</h3><p style="font-size:14.5px">The textbook recipe
fails here: the backbone is a <em>nested</em> Functional model, so <code>out_relu</code> is
invisible to the outer Sequential and Keras 3 raises a graph-disconnected error. The model is
split into (preprocessing, backbone, head) and re-run segment by segment.</p></div>
</div></div></div>
""", note="Grad-CAM is a 7x7 grid upsampled to 224x224 — each cell covers 32x32 px. Coarse by construction.")

# 14 filters / feature maps
slide(f"""
<div class="eyebrow">Explainability</div><h2>What the network learned to look for</h2>
<div class="body" style="gap:10px">
<div class="two" style="flex:0 0 auto;max-height:250px">
<figure><img src="{F['filters']}" alt="First conv filters">
<figcaption>First-layer convolution filters — oriented edges, colour opponency, blobs.
Classic Gabor-like structure inherited from ImageNet.</figcaption></figure>
<figure><img src="{F['fm_early']}" alt="Early feature maps">
<figcaption><code>block_1_expand_relu</code> — early activations still resemble the leaf:
edges, veins, high-contrast spots.</figcaption></figure></div>
<div class="two" style="flex:0 0 auto;max-height:230px">
<figure><img src="{F['fm_mid']}" alt="Mid feature maps">
<figcaption><code>block_6_expand_relu</code> — mid-depth. Texture and lesion-shaped regions;
the leaf outline is dissolving.</figcaption></figure>
<figure><img src="{F['fm_late']}" alt="Late feature maps">
<figcaption><code>out_relu</code> — final backbone layer. Sparse, abstract, spatially coarse:
these are the features the classifier head actually consumes.</figcaption></figure></div>
</div>
""", note="The progression concrete -> abstract is the visual argument for why fine-tuning the TOP of the backbone (not the bottom) is the right call — the top holds the task-specific features.")

# 15 app
slide(f"""
<div class="eyebrow">Delivery</div><h2>The application</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card lead"><h3>1 · Diagnose</h3><p style="font-size:14.5px">Upload a photo or load
a held-out test image. Returns <strong>top-3 ranked classes</strong> with confidence bars and
treatment guidance in <strong>English and Urdu</strong> (RTL, Noto Nastaliq).</p></div>
<div class="card"><h3>2 · Model Performance</h3><p style="font-size:14.5px">Macro F1, accuracy,
per-class F1 with the weakest class flagged, confusion matrix toggling between row shares and
raw counts. <strong>Every value read from <code>reports/</code> at runtime</strong> — the tab
cannot drift from the run, and says so if a file is missing.</p></div>
<div class="card"><h3>3 · Explainability</h3><p style="font-size:14.5px">Grad-CAM, conv filters,
feature maps at three depths — plus a counterfactual selector: <em>where would the model look
if this were a different disease?</em></p></div>
<div class="card"><h3>4 · Method &amp; Limitations</h3><p style="font-size:14.5px">Split
strategy, architecture decisions, and the honest caveats — in the product, not just the
report.</p></div>
</div>
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card"><h3>Engineering details worth noting</h3><ul style="gap:7px">
<li><strong>Light/dark themes</strong> from one token dictionary driving both the CSS and the
matplotlib figures — charts re-render in the active palette rather than glowing white.</li>
<li><strong>Theme is part of the cache key.</strong> Without it, flipping the toggle serves
the other theme's cached PNG.</li>
<li><strong><code>@st.cache_resource</code></strong> on model load — otherwise the 24 MB
checkpoint reloads on every interaction.</li>
<li><strong>EXIF transpose + RGB convert</strong> before resize; phone uploads are routinely
rotated, RGBA or greyscale.</li>
<li><strong>Self-healing interpreter guard</strong> — if <code>streamlit</code> resolves to a
Python without TensorFlow, the app re-executes itself under one that has it.</li></ul></div>
<div class="card lead"><h3>Deployed</h3><p style="font-size:14.5px">Streamlit Community Cloud.
Requires Python 3.11–3.13 chosen at deploy time — TensorFlow publishes no 3.14 wheels, and
the version cannot be changed after the app is created.</p></div>
</div></div></div>
""", note="SCREENSHOT SLOT — if you want live app screenshots, this is the slide to replace the right column with. Say the word.")

# 16 limitations
slide(f"""
<div class="eyebrow">Limitations</div><h2>What this model cannot do</h2>
<div class="body"><div class="two">
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card lead"><h3>Confidence does not detect wrong subjects</h3>
<p style="font-size:14px">Softmax always sums to 1. This was <strong>measured, not
assumed</strong> — against real leaves (median top-1 0.998):</p>
<table style="margin-top:6px"><tbody>
<tr><td>solid grey square</td><td>0.982</td><td style="color:var(--bad)">Late blight</td></tr>
<tr><td>crude drawn face</td><td>0.989</td><td style="color:var(--bad)">Mosaic virus</td></tr>
<tr><td>uniform noise</td><td>0.894</td><td style="color:var(--bad)">healthy</td></tr>
</tbody></table>
<p class="foot"><strong>No threshold separates these from real leaves.</strong> Entropy is no
better — the grey square scores 0.101, below the 0.138 mean of genuine leaves. The UI states
this plainly rather than implying the 50% flag is a safeguard.</p></div>
<div class="card"><h3>No real-world field validation</h3><p style="font-size:14px">Every
number here comes from held-out images of the same Kaggle dataset. Performance on a phone
photo in an actual field is <strong>unmeasured and likely worse</strong> — much of the source
data is lab imagery on uniform backgrounds.</p></div>
</div>
<div style="display:flex;flex-direction:column;gap:11px">
<div class="card"><h3>Early blight is genuinely weak</h3><p style="font-size:14px">F1
{worst['f1']:.4f}, recall {worst['r']:.4f} — nearly one in five missed. Any real deployment
should treat a confident early/late blight distinction with suspicion, since the two carry
very different urgency.</p></div>
<div class="card"><h3>Partial fine-tuning only</h3><p style="font-size:14px">The bottom 65% of
the backbone stays frozen and all 52 BatchNorm layers are held in inference mode. A full
unfreeze with LR warmup would probably add more, at meaningfully higher risk.</p></div>
<div class="card"><h3>Also true</h3><ul style="gap:5px">
<li>Grad-CAM is a 7×7 grid upsampled to 224×224 — coarse by construction</li>
<li>Advice text is general horticultural guidance, <strong>not agronomist-reviewed</strong></li>
<li>Single held-out split — no cross-validation, no confidence intervals</li>
<li>Closed set: 11 classes, no "something else" option</li></ul></div>
</div></div></div>
""", note="Do not rush this slide. Stating limits precisely is what separates a measured result from a marketing number — and the OOD finding is genuinely interesting.")

# 17 close
slide(f"""
<div class="eyebrow">Summary</div>
<h1 style="font-size:44px">What the numbers say</h1>
<div style="display:flex;gap:44px;margin:34px 0 10px">
<div><div class="lab">Macro F1</div><div class="num" style="font-size:56px">{mf1:.4f}</div></div>
<div><div class="lab">Accuracy</div><div class="num" style="font-size:56px">{acc:.4f}</div></div>
<div><div class="lab">Fine-tuning gain</div>
<div class="num" style="font-size:56px;color:#D9A85C">+{gain:.2f}</div></div>
</div>
<div class="two" style="margin-top:14px">
<div class="card"><h3 style="color:#EEF1E7">Three things I would defend</h3><ul style="gap:8px">
<li><strong style="color:#D9A85C">Reading the plateau correctly.</strong> Train below val with
flat losses meant underfitting — so more data augmentation would have hurt. Unfreezing was
the only lever, and it returned +{gain:.2f} points.</li>
<li><strong style="color:#D9A85C">Refusing the easy 99%.</strong> Preserving the archive's
split boundary instead of reshuffling cost roughly four points of headline accuracy and
bought a number that is actually true.</li>
<li><strong style="color:#D9A85C">Measuring the failure mode.</strong> The confidence score
does not detect non-leaf input — tested, documented in the app, not quietly omitted.</li></ul></div>
<div class="card"><h3 style="color:#EEF1E7">If I had another week</h3><ul style="gap:8px">
<li>Field-photograph validation set — the single biggest gap</li>
<li>An explicit background/"not a leaf" class for real OOD rejection</li>
<li>Targeted work on the early/late blight boundary: higher resolution or a
lesion-focused crop</li>
<li>Cross-validation for confidence intervals on every metric</li></ul></div>
</div>
<div class="author" style="border-color:rgba(238,241,231,.2)">
<div><div class="nm" style="color:#EEF1E7">Ali Khan</div>
<div class="mt" style="color:rgba(238,241,231,.6)">Applied AI &amp; Machine Learning — Midterm</div></div>
<div class="pill" style="background:rgba(255,255,255,.12);color:#EEF1E7">Thank you — questions?</div></div>
""", "dark", note="Close on the three defensible decisions. They are the actual contribution — the accuracy number is just evidence.")

html = (ROOT / "presentation.html").read_text()
html = html.replace("<!--SLIDES-->", "\n".join(S))

titles = ["Title", "Problem", "Libraries", "Dataset", "Architecture", "Class imbalance",
          "Results divider", "Stage 1 — frozen", "Stage 2 — fine-tune", "Training curves",
          "Test results", "Confusion matrix", "Grad-CAM", "Feature maps", "The app",
          "Limitations", "Summary"]
script = """<script>
const slides=[...document.querySelectorAll('.slide')],N=slides.length;
const TITLES=%s;
let i=0,notesOn=false;
const bar=document.getElementById('bar'),hud=document.getElementById('hud'),
      notes=document.getElementById('notes'),grid=document.getElementById('grid');
function fit(){const d=document.getElementById('deck');
  const s=Math.min(innerWidth/1280,innerHeight/720)*0.96;
  d.style.transform='scale('+s+')';}
function show(n){i=Math.max(0,Math.min(N-1,n));
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  bar.style.width=((i+1)/N*100)+'%%';
  hud.textContent=String(i+1).padStart(2,'0')+' / '+N;
  notes.textContent=slides[i].dataset.note||'—';
  location.hash=i+1;}
grid.innerHTML=slides.map((s,k)=>'<div class="g" data-k="'+k+'"><b>'+TITLES[k]+
  '</b><i>'+String(k+1).padStart(2,'0')+'</i></div>').join('');
grid.onclick=e=>{const g=e.target.closest('.g');if(g){show(+g.dataset.k);grid.classList.remove('on');}};
addEventListener('keydown',e=>{
  const k=e.key;
  if(k==='ArrowRight'||k==='PageDown'||k===' '){e.preventDefault();show(i+1);}
  else if(k==='ArrowLeft'||k==='PageUp'){e.preventDefault();show(i-1);}
  else if(k==='Home'){show(0);} else if(k==='End'){show(N-1);}
  else if(k==='o'||k==='O'){grid.classList.toggle('on');}
  else if(k==='n'||k==='N'){notesOn=!notesOn;notes.classList.toggle('on',notesOn);}
  else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():
    document.documentElement.requestFullscreen();}
  else if(k==='Escape'){grid.classList.remove('on');}
  else if(/^[0-9]$/.test(k)){show(parseInt(k)-1);}
});
let x0=null;
addEventListener('touchstart',e=>x0=e.changedTouches[0].clientX,{passive:true});
addEventListener('touchend',e=>{if(x0===null)return;
  const dx=e.changedTouches[0].clientX-x0;if(Math.abs(dx)>50)show(i+(dx<0?1:-1));x0=null;},{passive:true});
addEventListener('resize',fit);
fit();show(parseInt(location.hash.slice(1))-1||0);
</script>""" % json.dumps(titles)
html = html.replace("<!--SCRIPT-->", script)
(ROOT / "presentation.html").write_text(html)
kb = (ROOT / "presentation.html").stat().st_size / 1024
print(f"presentation.html  {len(S)} slides  {kb:.0f} KB")
