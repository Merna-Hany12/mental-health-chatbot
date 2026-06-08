"""
Module 1 — Language Detection
==============================
Multi-class classifier using TF-IDF (char n-grams) + LinearSVC.
Trained on the Language Identification dataset from Hugging Face.

Usage:
    from modules.language_detector import LanguageDetector
    detector = LanguageDetector()
    result = detector.predict("I feel very anxious today")
    # → {"language": "English", "code": "en", "confidence": 0.97}
"""

from __future__ import annotations

import re
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "language_detection_pipeline.pkl"

# ISO code → display name  (used everywhere consistently)
LANG_NAMES: Dict[str, str] = {
    "ar": "Arabic",      "bg": "Bulgarian",  "de": "German",
    "el": "Greek",       "en": "English",    "es": "Spanish",
    "fr": "French",      "hi": "Hindi",      "it": "Italian",
    "ja": "Japanese",    "nl": "Dutch",      "pl": "Polish",
    "pt": "Portuguese",  "ru": "Russian",    "sw": "Swahili",
    "th": "Thai",        "tr": "Turkish",    "ur": "Urdu",
    "vi": "Vietnamese",  "zh": "Chinese",
}

# Confidence threshold — below this we return "unknown" instead of a wrong guess
CONFIDENCE_THRESHOLD = 0.65

_KEEP = re.compile(r"[^\w\s]", flags=re.UNICODE)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Minimal cleaning — keeps Unicode letters so char n-grams stay meaningful.
    Removing diacritics or lowercasing script-specific chars would hurt accuracy.
    """
    text = text.strip().lower()
    text = _KEEP.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── Model Building ─────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Character-level TF-IDF (2-4 grams) + LinearSVC wrapped in calibration.

    Why char n-grams?
      - Script-agnostic: works for Arabic, CJK, Devanagari, Latin, Cyrillic etc.
      - Captures morpheme patterns that distinguish similar languages (es/pt, de/nl)
      - Robust to unseen words and typos
    """
    word_tfidf = TfidfVectorizer(
        analyzer     = "word",
        ngram_range  = (1, 2),
        max_features = 100_000,
        sublinear_tf = True,
        min_df       = 2,
        lowercase    = True,
        strip_accents = None,   # keep diacritics — they are language signals
    )

    char_tfidf = TfidfVectorizer(
        analyzer     = "char_wb",   # word-boundary-aware char n-grams
        ngram_range  = (2, 4),
        max_features = 200_000,
        sublinear_tf = True,
        min_df       = 2,
        lowercase    = True,
        strip_accents = None,
    )

    from sklearn.pipeline import FeatureUnion
    features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])

    return Pipeline([
        ("features", features),
        ("clf", CalibratedClassifierCV(
            estimator  = LinearSVC(C=1.0, max_iter=2000, dual=True, random_state=42),
            cv         = 3,
            method     = "sigmoid",
        )),
    ])


# ── Training ───────────────────────────────────────────────────────────────────
def train(save: bool = True) -> Pipeline:
    """
    Load the Language Identification dataset, train the pipeline,
    optionally save to disk, and return the fitted pipeline.

    The dataset labels are already ISO codes ("en", "ar", etc.)
    We use them directly as y — no mapping needed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Loading papluca/language-identification …")
    ds_train = load_dataset("papluca/language-identification", split="train")
    ds_test  = load_dataset("papluca/language-identification", split="test")

    df_train = ds_train.to_pandas()
    df_test  = ds_test.to_pandas()

    # ── Filter to supported languages ────────────────────────────────────────
    supported = set(LANG_NAMES.keys())
    df_train = df_train[df_train["labels"].isin(supported)].copy()
    df_test  = df_test[df_test["labels"].isin(supported)].copy()

    # ── Preprocess ────────────────────────────────────────────────────────────
    df_train["text_clean"] = df_train["text"].apply(preprocess)
    df_test["text_clean"]  = df_test["text"].apply(preprocess)

    X_train, y_train = df_train["text_clean"].tolist(), df_train["labels"].tolist()
    X_test,  y_test  = df_test["text_clean"].tolist(),  df_test["labels"].tolist()

    logger.info(f"Train : {len(X_train):,} samples | {len(set(y_train))} languages")
    logger.info(f"Test  : {len(X_test):,}  samples")

    # ── Fit ───────────────────────────────────────────────────────────────────
    pipeline = build_pipeline()
    logger.info("Fitting pipeline …")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    logger.info(f"\nTest Accuracy: {acc*100:.4f}%")
    logger.info("\n" + classification_report(
        y_test, y_pred,
        target_names=[LANG_NAMES[c] for c in sorted(set(y_test))],
        digits=4,
    ))

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Model saved → {MODEL_PATH}")

    return pipeline


# ── Detector Class ─────────────────────────────────────────────────────────────
class LanguageDetector:
    """
    High-level wrapper around the TF-IDF language classifier.
    Auto-loads a pre-trained model from disk; trains one if not found.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self._pipeline: Pipeline | None = None
        self._model_path = model_path
        self._load_or_train()

    # ── Private ───────────────────────────────────────────────────────────────
    def _load_or_train(self) -> None:
        if self._model_path.exists():
            logger.info(f"Loading language detector from {self._model_path}")
            self._pipeline = joblib.load(self._model_path)
        else:
            logger.warning("No saved model found — training from scratch …")
            self._pipeline = train(save=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str, threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
        """
        Classify the language of *text*.

        Parameters
        ----------
        text      : input string (any language)
        threshold : minimum confidence to trust the prediction.
                    Below this, returns code="en" with flag low_confidence=True.

        Returns
        -------
        {
            "language":       "English",
            "code":           "en",
            "confidence":     0.97,
            "low_confidence": False,
            "top5": [
                {"code": "en", "language": "English",  "prob": 0.97},
                {"code": "fr", "language": "French",   "prob": 0.01},
                ...
            ]
        }
        """
        # ── Edge case: empty input ────────────────────────────────────────────
        if not text or not text.strip():
            return {
                "language": "English", "code": "en",
                "confidence": 0.0, "low_confidence": True, "top5": [],
            }

        cleaned = preprocess(text)

        # pipeline.classes_ = sorted ISO codes e.g. ["ar","bg","de",...]
        # predict_proba returns probabilities in the same order
        proba   = self._pipeline.predict_proba([cleaned])[0]
        classes = self._pipeline.classes_           # ISO codes

        # ── Build sorted probability list ────────────────────────────────────
        pairs = sorted(
            zip(classes, proba),
            key=lambda x: -x[1]
        )
        top_code, top_prob = pairs[0]

        # ── Confidence check ─────────────────────────────────────────────────
        low_confidence = bool(top_prob < threshold)
        if low_confidence:
            # Don't return a wrong confident answer — flag it
            top_code = "en"

        return {
            "language":       LANG_NAMES.get(top_code, top_code),
            "code":           top_code,
            "confidence":     round(float(top_prob), 4),
            "low_confidence": low_confidence,
            "top5": [
                {
                    "code":     code,
                    "language": LANG_NAMES.get(code, code),
                    "prob":     round(float(prob), 4),
                }
                for code, prob in pairs[:5]
            ],
        }

    def batch_predict(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Predict language for a list of texts."""
        return [self.predict(t) for t in texts]


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train(save=True)
    else:
        detector = LanguageDetector()
        samples = [
            ("en", "my name is Merna"),
            ("en", "Thnak you"),
            ("ar", "السلام عليكم"),
            ("ar", "ازيك"),

            ("ar", "أشعر بالقلق الشديد ولا أستطيع النوم."),
            ("fr", "Je me sens très anxieux et je ne peux pas dormir."),
            ("es", "Me siento muy ansioso y no puedo dormir."),
            ("en", "I'm having trouble sleeping"),
            ("de", "Ich fühle mich sehr gestresst."),
            ("hi", "मुझे बहुत चिंता हो रही है।"),
            ("zh", "我最近感到非常焦虑。"),
            ("en", "hi"),
    ("en", "help me"),
    ("en", "I feel sad"),
    ("en", "my name is Merna"),
    ("en", "I cannot sleep at night"),

    # ── Arabic ───────────────────────────────────────────────
    ("ar", "مرحبا"),
    ("ar", "ساعدني"),
    ("ar", "أنا حزين"),
    ("ar", "اسمي مرنا"),
    ("ar", "لا أستطيع النوم"),

    # ── French ───────────────────────────────────────────────
    ("fr", "bonjour"),
    ("fr", "aide moi"),
    ("fr", "je suis triste"),
    ("fr", "je m'appelle Merna"),
    ("fr", "je ne peux pas dormir"),

    # ── Spanish ──────────────────────────────────────────────
    ("es", "hola"),
    ("es", "ayúdame"),
    ("es", "me siento triste"),
    ("es", "me llamo Merna"),
    ("es", "no puedo dormir"),

    # ── German ───────────────────────────────────────────────
    ("de", "hallo"),
    ("de", "hilf mir"),
    ("de", "ich bin traurig"),
    ("de", "ich heiße Merna"),
    ("de", "ich kann nicht schlafen"),

    # ── Italian ──────────────────────────────────────────────
    ("it", "ciao"),
    ("it", "aiutami"),
    ("it", "mi sento triste"),
    ("it", "mi chiamo Merna"),
    ("it", "non riesco a dormire"),

    # ── Portuguese ───────────────────────────────────────────
    ("pt", "olá"),
    ("pt", "me ajuda"),
    ("pt", "estou triste"),
    ("pt", "meu nome é Merna"),
    ("pt", "não consigo dormir"),

    # ── Russian ──────────────────────────────────────────────
    ("ru", "привет"),
    ("ru", "помоги мне"),
    ("ru", "мне грустно"),
    ("ru", "меня зовут Мерна"),
    ("ru", "я не могу спать"),

    # ── Hindi ────────────────────────────────────────────────
    ("hi", "नमस्ते"),
    ("hi", "मेरी मदद करो"),
    ("hi", "मैं दुखी हूं"),
    ("hi", "मेरा नाम मेर्ना है"),
    ("hi", "मुझे नींद नहीं आती"),

    # ── Chinese ──────────────────────────────────────────────
    ("zh", "你好"),
    ("zh", "帮帮我"),
    ("zh", "我很难过"),
    ("zh", "我叫Merna"),
    ("zh", "我睡不着"),

    # ── Japanese ─────────────────────────────────────────────
    ("ja", "こんにちは"),
    ("ja", "助けて"),
    ("ja", "悲しい"),
    ("ja", "私の名前はメルナです"),
    ("ja", "眠れません"),

    # ── Turkish ──────────────────────────────────────────────
    ("tr", "merhaba"),
    ("tr", "bana yardım et"),
    ("tr", "üzgünüm"),
    ("tr", "benim adım Merna"),
    ("tr", "uyuyamıyorum"),

    # ── Dutch ────────────────────────────────────────────────
    ("nl", "hallo"),
    ("nl", "help me"),
    ("nl", "ik ben verdrietig"),
    ("nl", "ik kan niet slapen"),

    # ── Polish ───────────────────────────────────────────────
    ("pl", "cześć"),
    ("pl", "jestem smutny"),
    ("pl", "mam na imię Merna"),
    ("pl", "nie mogę spać"),

    # ── Swedish (sw = Swahili in your model) ─────────────────
    ("sw", "habari"),
    ("sw", "nisaidie"),
    ("sw", "nina huzuni"),
    ("sw", "jina langu ni Merna"),
    ("sw", "siwezi kulala"),

    # ── Bulgarian ────────────────────────────────────────────
    ("bg", "здравей"),
    ("bg", "помогни ми"),
    ("bg", "тъжен съм"),
    ("bg", "казвам се Мерна"),
    ("bg", "не мога да спя"),

    # ── Greek ────────────────────────────────────────────────
    ("el", "γεια"),
    ("el", "βοήθησέ με"),
    ("el", "είμαι λυπημένος"),
    ("el", "με λένε Μέρνα"),
    ("el", "δεν μπορώ να κοιμηθώ"),

    # ── Thai ─────────────────────────────────────────────────
    ("th", "สวัสดี"),
    ("th", "ช่วยฉันด้วย"),
    ("th", "ฉันเศร้า"),
    ("th", "ชื่อของฉันคือเมอร์นา"),
    ("th", "ฉันนอนไม่หลับ"),

    # ── Urdu ─────────────────────────────────────────────────
    ("ur", "ہیلو"),
    ("ur", "میری مدد کرو"),
    ("ur", "میں اداس ہوں"),
    ("ur", "میرا نام مرنا ہے"),
    ("ur", "مجھے نیند نہیں آتی"),

    # ── Vietnamese ───────────────────────────────────────────
    ("vi", "xin chào"),
    ("vi", "giúp tôi với"),
    ("vi", "tôi buồn"),
    ("vi", "tên tôi là Merna"),
    ("vi", "tôi không ngủ được"),

        ]

        print("\n=== Language Detection Test ===")
        correct = 0
        for true_code, text in samples:
            r      = detector.predict(text)
            status = "✅" if r["code"] == true_code else "❌"
            flag   = " ⚠️ low confidence" if r["low_confidence"] else ""
            print(f"  {status} [{r['code']}] {r['language']:<14} ({r['confidence']:.2%}){flag}")
            print(f"       True: [{true_code}]  Input: {text[:60]}")
            if r["low_confidence"]:
                print(f"       Top guess was: [{r['top5'][0]['code']}] {r['top5'][0]['prob']:.2%}")
            correct += (r["code"] == true_code)

        print(f"\n  Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.0f}%)")