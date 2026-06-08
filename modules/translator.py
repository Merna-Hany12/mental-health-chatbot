"""
Translator Module
==================
Translates user query to English before passing to Emotion Classifier.
Uses deep_translator (Google Translate) — FREE, no API key needed.

Install:
    pip install deep-translator
    uv add deep-translator

Pipeline position:
    User message (Arabic/French/etc)
        │
        ▼  Module 1: Language Detection → "Arabic", "ar"
        │
        ▼  Translator (this module) → English text
        │
        ▼  Module 2: Emotion Classifier (English input) → correct emotion

Usage:
    from modules.translator import Translator
    t = Translator()
    result = t.translate("أشعر بالقلق الشديد", source_lang_code="ar")
    # → {"translated": "I feel very anxious", "was_translated": True, ...}
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Languages that don't need translation
ENGLISH_CODES = {"en", "english"}

# deep_translator language code map
# Maps our LanguageDetector output → deep_translator code
LANG_CODE_MAP: Dict[str, str] = {
    "ar": "ar",   # Arabic
    "fr": "fr",   # French
    "es": "es",   # Spanish
    "de": "de",   # German
    "it": "it",   # Italian
    "pt": "pt",   # Portuguese
    "ru": "ru",   # Russian
    "tr": "tr",   # Turkish
    "hi": "hi",   # Hindi
    "en": "en",   # English
}


class Translator:
    """
    Translates any language to English using deep_translator (Google Translate).
    No API key required — completely free.
    Falls back to original text on any error.
    """

    def __init__(self):
        self._check_dependency()

    def _check_dependency(self) -> None:
        try:
            from deep_translator import GoogleTranslator
            # Quick test to verify it works
            GoogleTranslator(source="auto", target="en")
            logger.info("✅ Translator ready (deep_translator / Google Translate)")
        except ImportError:
            logger.warning(
                "deep_translator not installed. "
                "Run: pip install deep-translator\n"
                "Translator will return original text until installed."
            )
        except Exception as e:
            logger.warning(f"Translator init warning: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────
    def translate(
        self,
        text:             str,
        source_lang:      str = "unknown",   # display name e.g. "Arabic"
        source_lang_code: str = "auto",      # ISO code e.g. "ar"
    ) -> Dict[str, Any]:
        """
        Translate text to English.

        Parameters
        ----------
        text             : The user's original message.
        source_lang      : Language display name from LanguageDetector e.g. "Arabic".
        source_lang_code : ISO code from LanguageDetector e.g. "ar".

        Returns
        -------
        {
            "translated":     "I feel very anxious",
            "original":       "أشعر بالقلق الشديد",
            "source_lang":    "Arabic",
            "was_translated": True
        }
        """
        if not text or not text.strip():
            return {
                "translated":     text,
                "original":       text,
                "source_lang":    source_lang,
                "was_translated": False,
            }

        # Skip if already English
        if source_lang_code.lower() in ENGLISH_CODES or source_lang.lower() in ENGLISH_CODES:
            logger.info("Translation skipped — already English.")
            return {
                "translated":     text,
                "original":       text,
                "source_lang":    source_lang,
                "was_translated": False,
            }

        try:
            from deep_translator import GoogleTranslator

            # Use "auto" if code not in our map
            src_code = LANG_CODE_MAP.get(source_lang_code.lower(), "auto")

            translated = GoogleTranslator(
                source = src_code,
                target = "en",
            ).translate(text)

            logger.info(
                f"Translated [{source_lang}] → [English]: {translated[:80]}"
            )

            return {
                "translated":     translated,
                "original":       text,
                "source_lang":    source_lang,
                "was_translated": True,
            }

        except ImportError:
            logger.error("deep_translator not installed. Run: pip install deep-translator")
            return {
                "translated":     text,
                "original":       text,
                "source_lang":    source_lang,
                "was_translated": False,
            }

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "translated":     text,   # fallback to original
                "original":       text,
                "source_lang":    source_lang,
                "was_translated": False,
            }

    def batch_translate(
        self,
        texts:            list[str],
        source_lang:      str = "unknown",
        source_lang_code: str = "auto",
    ) -> list[Dict[str, Any]]:
        """Translate a list of texts."""
        return [
            self.translate(t, source_lang=source_lang, source_lang_code=source_lang_code)
            for t in texts
        ]

    def translate_from_english(
        self,
        text:             str,
        target_lang:      str = "English",
        target_lang_code: str = "en",
    ) -> Dict[str, Any]:
        """
        Translate an English response to the user's detected language.

        Falls back to the original English text if translation is skipped,
        unsupported, or temporarily unavailable.
        """
        if not text or not text.strip():
            return {
                "translated":     text,
                "original":       text,
                "target_lang":    target_lang,
                "was_translated": False,
            }

        if target_lang_code.lower() in ENGLISH_CODES or target_lang.lower() in ENGLISH_CODES:
            logger.info("Response translation skipped - target language is English.")
            return {
                "translated":     text,
                "original":       text,
                "target_lang":    target_lang,
                "was_translated": False,
            }

        try:
            from deep_translator import GoogleTranslator

            target_code = LANG_CODE_MAP.get(target_lang_code.lower())
            if target_code is None:
                logger.warning(f"Response translation skipped - unsupported target language: {target_lang}")
                return {
                    "translated":     text,
                    "original":       text,
                    "target_lang":    target_lang,
                    "was_translated": False,
                }

            translated = GoogleTranslator(
                source = "en",
                target = target_code,
            ).translate(text)

            logger.info(
                f"Translated direct response [English] -> [{target_lang}]: {translated[:80]}"
            )

            return {
                "translated":     translated,
                "original":       text,
                "target_lang":    target_lang,
                "was_translated": True,
            }

        except ImportError:
            logger.error("deep_translator not installed. Run: pip install deep-translator")
            return {
                "translated":     text,
                "original":       text,
                "target_lang":    target_lang,
                "was_translated": False,
            }

        except Exception as e:
            logger.error(f"Response translation failed: {e}")
            return {
                "translated":     text,
                "original":       text,
                "target_lang":    target_lang,
                "was_translated": False,
            }


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    t = Translator()

    samples = [
        ("أشعر بالقلق الشديد ولا أستطيع النوم.", "Arabic",     "ar"),
        ("Je me sens très déprimé et sans espoir.", "French",   "fr"),
        ("Me siento muy ansioso y no puedo dormir.", "Spanish", "es"),
        ("Ich fühle mich sehr ängstlich.",           "German",  "de"),
        ("I feel very anxious and cannot sleep.",    "English", "en"),
    ]

    print("\n=== Translation Test (deep_translator) ===")
    for text, lang, code in samples:
        result = t.translate(text, source_lang=lang, source_lang_code=code)
        status = "✅ translated" if result["was_translated"] else "⏭ skipped"
        print(f"\n  [{lang}] {status}")
        print(f"  Original:   {result['original']}")
        print(f"  Translated: {result['translated']}")