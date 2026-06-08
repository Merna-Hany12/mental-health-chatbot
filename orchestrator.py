import os
import logging
import random
import time
from typing import Dict, Any

from modules.language_detector  import LanguageDetector
from modules.emotion_classifier import EmotionClassifier
from modules.intent_classifier  import IntentClassifier
from modules.translator         import Translator
from modules.rag_pipeline       import RAGPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIRECT_RESPONSES = {
    "greeting": [
        "Hello! I'm here to listen. Feel free to share what's on your mind. How are you feeling today?",
        "Hi there. I'm glad you reached out. What's been on your mind lately?",
        "Hello, I'm here with you. How can I support you today?",
    ],
    "goodbye": [
        "Take care of yourself. I'm here whenever you need to talk.",
        "Wishing you a calm rest of your day. You can come back whenever you need support.",
        "Goodbye for now. Be gentle with yourself, and reach out again anytime.",
    ],
    "gratitude": [
        "You're welcome. I'm really glad I could help. Remember, reaching out is always a sign of strength.",
        "I'm glad that helped. You deserve support, and I'm here whenever you want to talk.",
        "You're very welcome. I'm here anytime you need a steady place to share what's going on.",
    ],
    "out_of_scope": [
        "I'm a mental health support assistant, so I'm best suited for questions about emotions, stress, anxiety, and wellbeing. Is there something along those lines I can help with?",
        "I can help most with mental health and emotional wellbeing topics. If something is weighing on you, I'm here to listen.",
        "That is outside what I can support well, but I can help with feelings, coping strategies, stress, anxiety, and wellbeing.",
    ],
}


class Orchestrator:
    def __init__(self, groq_api_key=None, qdrant_url=None, qdrant_api_key=None):
        self.groq_api_key   = groq_api_key   or os.getenv("groq_api_key", "")
        self.qdrant_url     = qdrant_url     or os.getenv("qdrant_url", "")
        self.qdrant_api_key = qdrant_api_key or os.getenv("qdrant_api_key", "")

        self.lang_detector = None
        self.translator    = None
        self.emotion_clf   = None
        self.intent_clf    = None
        self.rag           = None
        self._ready        = False

    def _setup(self):
        if self._ready:
            return

        logger.info("Loading modules...")

        self.lang_detector = LanguageDetector()
        self.translator    = Translator()
        self.emotion_clf   = EmotionClassifier()
        self.intent_clf    = IntentClassifier(groq_api_key=self.groq_api_key)

        if self.qdrant_url and self.qdrant_api_key and self.groq_api_key:
            self.rag = RAGPipeline(
                qdrant_url     = self.qdrant_url,
                qdrant_api_key = self.qdrant_api_key,
                groq_api_key   = self.groq_api_key,
            )
        else:
            logger.warning("RAG pipeline skipped - missing credentials in .env")

        self._ready = True
        logger.info("All modules loaded.")

    def warm_up(self):
        """Load all modules before the first chat request."""
        self._setup()

    def chat(self, message: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        logger.info("Chat request received.")
        self._setup()

        if not message or not message.strip():
            logger.info("Chat request is empty; returning fallback response.")
            return self._empty_response()

        # step 1 - detect language
        logger.info("Step 1/6: Detecting language...")
        lang = self.lang_detector.predict(message)
        logger.info(f"Language: {lang['language']} ({lang['confidence']:.0%})")

        # step 2 - translate to english (needed for emotion classifier)
        logger.info("Step 2/6: Translating message to English if needed...")
        translation = self.translator.translate(
            text             = message,
            source_lang      = lang["language"],
            source_lang_code = lang["code"],
        )
        english_text = translation["translated"]
        logger.info(f"Translation complete. Was translated: {translation.get('was_translated', False)}")

        # step 3 - classify intent (on original message, groq handles all languages)
        logger.info("Step 3/6: Classifying intent...")
        intent_result = self.intent_clf.predict(message)
        intent        = intent_result["intent"]
        logger.info(f"Intent: {intent}")

        # step 4 - if not a mental health question, reply directly without RAG
        logger.info("Step 4/6: Choosing response route...")
        if intent != "asking_mental_health_question":
            logger.info("Route: direct response, RAG skipped.")
            answer = self._direct_response(intent, lang)
            logger.info(f"Chat request complete in {(time.perf_counter() - t0) * 1000:.1f} ms.")
            return {
                "answer":      answer,
                "language":    lang,
                "translation": translation,
                "intent":      intent_result,
                "emotion":     None,
                "sources":     [],
                "used_rag":    False,
            }

        # step 5 - classify emotion on the english version
        logger.info("Step 5/6: Classifying emotion...")
        emotion = self.emotion_clf.predict(english_text)
        logger.info(f"Emotion: {emotion['emotion']} ({emotion['confidence']:.0%})")

        # step 6 - generate answer with RAG
        logger.info("Step 6/6: Retrieving sources and generating RAG answer...")
        if self.rag is None:
            logger.warning("RAG route selected, but RAG pipeline is not configured.")
            logger.info(f"Chat request complete in {(time.perf_counter() - t0) * 1000:.1f} ms.")
            return {
                "answer":      "RAG pipeline is not configured. Please check your .env file.",
                "language":    lang,
                "translation": translation,
                "intent":      intent_result,
                "emotion":     emotion,
                "sources":     [],
                "used_rag":    False,
            }

        result = self.rag.ask(
            question      = english_text,
            emotion_tone  = emotion.get("tone", "Be warm and empathetic."),
            emotion_label = emotion.get("emotion", "neutral"),
            language      = lang["language"],
            language_code = lang["code"],
        )
        logger.info(f"RAG answer generated with {len(result.get('sources', []))} sources.")
        logger.info(f"Chat request complete in {(time.perf_counter() - t0) * 1000:.1f} ms.")

        return {
            "answer":      result["answer"],
            "language":    lang,
            "translation": translation,
            "intent":      intent_result,
            "emotion":     emotion,
            "sources":     result.get("sources", []),
            "used_rag":    True,
        }

    def _direct_response(self, intent: str, lang: Dict[str, Any]) -> str:
        responses = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])
        response = random.choice(responses)

        translated = self.translator.translate_from_english(
            text             = response,
            target_lang      = lang["language"],
            target_lang_code = lang["code"],
        )
        return translated["translated"]

    def _empty_response(self):
        return {
            "answer":      "Please share what's on your mind.",
            "language":    {"language": "English", "code": "en", "confidence": 0.0},
            "translation": {"translated": "", "was_translated": False},
            "intent":      {"intent": "greeting", "confidence": "low"},
            "emotion":     None,
            "sources":     [],
            "used_rag":    False,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "language_detector":  self.lang_detector is not None,
            "translator":         self.translator    is not None,
            "emotion_classifier": self.emotion_clf   is not None,
            "intent_classifier":  self.intent_clf    is not None,
            "rag_pipeline":       self.rag           is not None,
            "ready":              self._ready,
        }
