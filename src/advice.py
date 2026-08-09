"""Static treatment advice per class, in English and Urdu.

Lookup is NORMALISED rather than exact-match: keys are compared after lowercasing and
stripping every non-alphanumeric character, so "Tomato___Late_blight", "Late blight" and
"late-blight" all resolve to the same entry. Each entry also carries `aliases` for names
that do not normalise together (e.g. the PlantVillage "Tomato___" prefixes, or
"Spider_mites Two-spotted_spider_mite").

This exists because the model is trained in Colab and served locally: the class names
come from whatever folder names the archive shipped, and a mismatch between those and
these keys would otherwise show a blank card with no error. get_advice() instead returns
an explicit FALLBACK entry, and evaluate_model.py warns at startup about any class that
hits it.

The advice text is general horticultural guidance for a course project. It has not been
reviewed by an agronomist and is not a substitute for local extension-service advice --
say so in the UI and the README.
"""

from __future__ import annotations

import re

_NORMALISE = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> str:
    """'Tomato___Late_blight' -> 'tomatolateblight'"""
    return _NORMALISE.sub("", name.lower())


# label: shown in the UI instead of the raw folder name.
# aliases: alternate spellings seen across tomato datasets.
ADVICE: dict[str, dict] = {
    "Bacterial_spot": {
        "label": "Bacterial Spot",
        "aliases": ["Tomato___Bacterial_spot", "bacterial spot"],
        "en": (
            "Remove and destroy infected leaves as soon as you spot them. Apply a "
            "copper-based spray weekly while symptoms persist. Water at the base rather "
            "than overhead — splashing spreads the bacteria. Use certified disease-free "
            "seed and avoid planting tomatoes in the same bed next season."
        ),
        "ur": (
            "متاثرہ پتے فوراً توڑ کر تلف کر دیں۔ علامات ختم ہونے تک ہفتہ وار کاپر پر مبنی "
            "اسپرے کریں۔ پودوں کو اوپر سے پانی نہ دیں بلکہ جڑ میں دیں، چھینٹوں سے جراثیم "
            "پھیلتے ہیں۔ تصدیق شدہ صحت مند بیج استعمال کریں اور اگلے سیزن میں اسی جگہ "
            "ٹماٹر نہ لگائیں۔"
        ),
    },
    "Early_blight": {
        "label": "Early Blight",
        "aliases": ["Tomato___Early_blight", "early blight"],
        "en": (
            "Pick off the affected lower leaves and dispose of them away from the plot. "
            "Mulch the soil surface so rain cannot splash spores upward. Spray mancozeb "
            "or chlorothalonil at label rates. Stake the plants and thin dense growth to "
            "keep air moving through the canopy."
        ),
        "ur": (
            "نیچے والے متاثرہ پتے توڑ کر کھیت سے دور تلف کر دیں۔ زمین پر بھوسے یا گھاس کی "
            "تہہ ڈالیں تاکہ بارش کے چھینٹوں سے جراثیم اوپر نہ جائیں۔ مینکوزیب یا "
            "کلوروتھیلونل ہدایت کے مطابق اسپرے کریں۔ پودوں کو سہارا دیں اور گھنی شاخیں "
            "چھانٹ کر ہوا کا گزر بہتر بنائیں۔"
        ),
    },
    "Late_blight": {
        "label": "Late Blight",
        "aliases": ["Tomato___Late_blight", "late blight"],
        "en": (
            "Act immediately — this spreads through a crop in days in cool, wet weather. "
            "Pull out infected plants entirely and burn or bag them; do not compost. "
            "Protect the remaining healthy plants with copper oxychloride. Stop overhead "
            "irrigation and water early in the day so foliage dries fast."
        ),
        "ur": (
            "فوری کارروائی کریں — ٹھنڈے اور مرطوب موسم میں یہ بیماری چند دنوں میں پوری "
            "فصل میں پھیل جاتی ہے۔ متاثرہ پودے جڑ سمیت اکھاڑ کر جلا دیں یا تھیلے میں بند "
            "کر کے پھینک دیں، کھاد میں ہرگز نہ ڈالیں۔ باقی صحت مند پودوں پر کاپر آکسی "
            "کلورائیڈ اسپرے کریں۔ اوپر سے پانی دینا بند کریں اور صبح سویرے پانی دیں تاکہ "
            "پتے جلد خشک ہو جائیں۔"
        ),
    },
    "Leaf_Mold": {
        "label": "Leaf Mould",
        "aliases": ["Tomato___Leaf_Mold", "leaf mold", "leaf mould"],
        "en": (
            "Mostly a tunnel and greenhouse problem driven by humidity. Increase "
            "ventilation, widen plant spacing, and keep night-time humidity down. Remove "
            "affected leaves. Apply a labelled fungicide if it keeps spreading, and "
            "choose resistant varieties next season."
        ),
        "ur": (
            "یہ زیادہ تر ٹنل اور گرین ہاؤس کا مسئلہ ہے جو نمی سے بڑھتا ہے۔ ہوا کی "
            "آمدورفت بڑھائیں، پودوں کے درمیان فاصلہ زیادہ رکھیں اور رات کی نمی کم کریں۔ "
            "متاثرہ پتے توڑ دیں۔ پھیلاؤ جاری رہے تو منظور شدہ پھپھوندی کش دوا استعمال "
            "کریں، اور اگلے سیزن مزاحم قسم کاشت کریں۔"
        ),
    },
    "Septoria_leaf_spot": {
        "label": "Septoria Leaf Spot",
        "aliases": ["Tomato___Septoria_leaf_spot", "septoria leaf spot", "septoria"],
        "en": (
            "Strip the infected lower leaves and destroy them. Mulch to break the "
            "soil-to-leaf splash cycle. Keep water off the foliage. Spray mancozeb or a "
            "copper fungicide every 7–10 days in wet spells. Clear all crop debris at the "
            "end of the season — the fungus overwinters in it."
        ),
        "ur": (
            "نیچے کے متاثرہ پتے اتار کر تلف کر دیں۔ زمین پر ملچ ڈالیں تاکہ مٹی سے پتوں تک "
            "چھینٹوں کا سلسلہ ٹوٹ جائے۔ پتوں پر پانی نہ پڑنے دیں۔ بارشوں کے دنوں میں ہر "
            "7 سے 10 دن بعد مینکوزیب یا کاپر پھپھوندی کش اسپرے کریں۔ سیزن کے آخر میں فصل "
            "کی باقیات مکمل صاف کریں، پھپھوندی انہی میں سردیاں گزارتی ہے۔"
        ),
    },
    "Spider_mites": {
        "label": "Spider Mites (Two-Spotted)",
        "aliases": [
            "Spider_mites Two-spotted_spider_mite",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "two spotted spider mite",
            "spider mite",
        ],
        "en": (
            "This is a pest, not a disease — look for fine webbing and stippled leaves. "
            "Hose the undersides of leaves with a strong jet of water to knock mites off. "
            "Follow with insecticidal soap or neem oil, repeating every 5–7 days. Keep "
            "plants well watered; mite populations explode on drought-stressed plants."
        ),
        "ur": (
            "یہ بیماری نہیں بلکہ ایک کیڑا ہے — پتوں پر باریک جالا اور چھوٹے دھبے دیکھیں۔ "
            "پتوں کی نچلی سطح پر تیز پانی کا چھڑکاؤ کریں تاکہ کیڑے جھڑ جائیں۔ اس کے بعد "
            "کیڑے مار صابن یا نیم کا تیل ہر 5 سے 7 دن بعد استعمال کریں۔ پودوں کو پانی کی "
            "کمی نہ ہونے دیں، خشکی میں یہ کیڑا بہت تیزی سے بڑھتا ہے۔"
        ),
    },
    "Target_Spot": {
        "label": "Target Spot",
        "aliases": ["Tomato___Target_Spot", "target spot"],
        "en": (
            "Remove affected leaves and any spotted fruit. Open up the canopy and widen "
            "spacing so leaves dry quickly after rain or dew. Apply a labelled fungicide "
            "such as chlorothalonil. Rotate away from tomatoes and peppers for two "
            "seasons."
        ),
        "ur": (
            "متاثرہ پتے اور داغ دار پھل ہٹا دیں۔ شاخیں چھانٹ کر اور فاصلہ بڑھا کر پودوں "
            "کو کھلا کریں تاکہ بارش یا شبنم کے بعد پتے جلد خشک ہوں۔ کلوروتھیلونل جیسی "
            "منظور شدہ پھپھوندی کش دوا اسپرے کریں۔ دو سیزن تک اس جگہ ٹماٹر اور مرچ کاشت "
            "نہ کریں۔"
        ),
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "label": "Yellow Leaf Curl Virus",
        "aliases": [
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "yellow leaf curl virus",
            "tylcv",
        ],
        "en": (
            "There is no cure once a plant is infected. Pull out and destroy affected "
            "plants so they stop acting as a reservoir. The virus is carried by "
            "whiteflies, so control is really whitefly control: yellow sticky traps, "
            "reflective mulch, and insecticidal soap on the undersides of leaves. Plant "
            "TYLCV-resistant varieties next season."
        ),
        "ur": (
            "پودے کے متاثر ہونے کے بعد اس کا کوئی علاج نہیں۔ متاثرہ پودے اکھاڑ کر تلف کر "
            "دیں تاکہ وہ وائرس کا ذخیرہ نہ بنیں۔ یہ وائرس سفید مکھی پھیلاتی ہے، اس لیے "
            "اصل علاج سفید مکھی کا کنٹرول ہے: زرد چپکنے والے کارڈ لگائیں، چمکدار ملچ "
            "استعمال کریں اور پتوں کی نچلی سطح پر کیڑے مار صابن لگائیں۔ اگلے سیزن مزاحمت "
            "رکھنے والی اقسام کاشت کریں۔"
        ),
    },
    "Tomato_mosaic_virus": {
        "label": "Mosaic Virus",
        "aliases": [
            "Tomato___Tomato_mosaic_virus",
            "tomato mosaic virus",
            "mosaic virus",
            "tmv",
        ],
        "en": (
            "No chemical treatment works. Remove infected plants and wash your hands and "
            "tools with soap before touching healthy ones — the virus spreads by contact. "
            "Do not use tobacco near the plants, as it can carry the virus. Choose "
            "resistant seed and avoid saving seed from an infected crop."
        ),
        "ur": (
            "اس کا کوئی کیمیائی علاج نہیں۔ متاثرہ پودے نکال دیں اور صحت مند پودوں کو ہاتھ "
            "لگانے سے پہلے ہاتھ اور اوزار صابن سے دھوئیں، یہ وائرس چھونے سے پھیلتا ہے۔ "
            "پودوں کے قریب تمباکو استعمال نہ کریں کیونکہ اس میں وائرس ہو سکتا ہے۔ مزاحم "
            "بیج استعمال کریں اور متاثرہ فصل سے بیج محفوظ نہ کریں۔"
        ),
    },
    "powdery_mildew": {
        "label": "Powdery Mildew",
        "aliases": ["Tomato___powdery_mildew", "powdery mildew"],
        "en": (
            "Look for a white, dusty coating on the upper leaf surface. Improve airflow "
            "and cut back crowded growth. Spray wettable sulphur or potassium "
            "bicarbonate; neem oil also works on light infections. Avoid heavy nitrogen "
            "feeding, which produces the soft new growth this fungus prefers."
        ),
        "ur": (
            "پتوں کی اوپری سطح پر سفید سفوف جیسی تہہ دیکھیں۔ ہوا کی آمدورفت بہتر کریں اور "
            "گھنی شاخیں چھانٹ دیں۔ گندھک (سلفر) یا پوٹاشیم بائی کاربونیٹ کا اسپرے کریں؛ "
            "ہلکے حملے میں نیم کا تیل بھی مؤثر ہے۔ نائٹروجن والی کھاد زیادہ نہ ڈالیں، اس "
            "سے نرم نئی شاخیں نکلتی ہیں جو یہ پھپھوندی پسند کرتی ہے۔"
        ),
    },
    "healthy": {
        "label": "Healthy",
        "aliases": ["Tomato___healthy", "Healthy"],
        "en": (
            "No disease signs in this leaf. Keep checking the undersides of leaves weekly "
            "— most problems are visible there first. Water at the base, feed a balanced "
            "fertiliser, and keep the bed clear of fallen debris."
        ),
        "ur": (
            "اس پتے میں بیماری کی کوئی علامت نہیں۔ ہفتہ وار پتوں کی نچلی سطح کا معائنہ "
            "جاری رکھیں، اکثر مسائل سب سے پہلے وہیں نظر آتے ہیں۔ جڑ میں پانی دیں، متوازن "
            "کھاد استعمال کریں اور کیاری سے گرے ہوئے پتے صاف رکھیں۔"
        ),
    },
}

FALLBACK: dict = {
    "label": None,  # filled in with the raw class name by get_advice()
    "en": (
        "No advice text has been written for this class yet. Add an entry for it in "
        "src/advice.py."
    ),
    "ur": "اس قسم کے لیے ابھی کوئی مشورہ درج نہیں کیا گیا۔",
}

# Built once: every normalised key and alias -> canonical entry.
_INDEX: dict[str, dict] = {}
for _key, _entry in ADVICE.items():
    _INDEX[normalise(_key)] = _entry
    for _alias in _entry.get("aliases", ()):
        _INDEX.setdefault(normalise(_alias), _entry)


def get_advice(class_name: str) -> dict:
    """Return {label, en, ur} for a class name, never raising on an unknown one.

    Falls back to a visible placeholder rather than a blank card, so a naming mismatch
    between the trained model and this file is obvious instead of silent.
    """
    entry = _INDEX.get(normalise(class_name))
    if entry is None:
        return {**FALLBACK, "label": prettify(class_name), "known": False}
    return {
        "label": entry.get("label") or prettify(class_name),
        "en": entry["en"],
        "ur": entry["ur"],
        "known": True,
    }


def prettify(class_name: str) -> str:
    """'Tomato___Late_blight' -> 'Late Blight', for display when there is no label."""
    cleaned = class_name.replace("___", " ").replace("_", " ").strip()
    cleaned = re.sub(r"^(?i:tomato)\s+", "", cleaned)
    return cleaned.title() if cleaned else class_name


def missing_advice(class_names: list[str]) -> list[str]:
    """Which of these class names have no advice entry. Used by evaluate_model.py."""
    return [name for name in class_names if normalise(name) not in _INDEX]
