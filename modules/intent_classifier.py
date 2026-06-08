"""
Module 3 — Intent Classifier
================================
Few-shot LLM prompting via Groq to classify user intent into:
    greeting | goodbye | gratitude | asking_mental_health_question | out_of_scope

Uses structured JSON output to ensure reliable parsing.
Falls back to rule-based classification if the API is unavailable.

Usage:
    from modules.intent_classifier import IntentClassifier
    clf = IntentClassifier(groq_api_key="...")
    result = clf.predict("I feel very depressed and cannot cope")
    # → {"intent": "asking_mental_health_question", "confidence": "high", "reasoning": "..."}
"""

from __future__ import annotations

import re
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_INTENTS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
}

INTENT_DESCRIPTIONS = {
    "greeting":                     "User is saying hello or starting the conversation.",
    "goodbye":                      "User is ending the conversation or saying farewell.",
    "gratitude":                    "User is expressing thanks or appreciation.",
    "asking_mental_health_question": "User is asking about mental health, emotions, therapy, "
                                     "coping strategies, depression, anxiety, stress, relationships, "
                                     "trauma, or any psychological concern.",
    "out_of_scope":                 "User is asking something unrelated to mental health "
                                     "(e.g. coding, geography, cooking, general knowledge).",
}

# ── Few-shot examples ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    # Greetings
    {"text": "Hello!",                                           "intent": "greeting"},
    {"text": "Hi there, good morning",                          "intent": "greeting"},
    {"text": "Hey, how are you?",                               "intent": "greeting"},
    # Goodbyes
    {"text": "Goodbye, talk to you later",                      "intent": "goodbye"},
    {"text": "Thanks, bye!",                                    "intent": "goodbye"},
    {"text": "See you soon",                                    "intent": "goodbye"},
    # Gratitude
    {"text": "Thank you so much for your help",                 "intent": "gratitude"},
    {"text": "I really appreciate your support",               "intent": "gratitude"},
    {"text": "That was very helpful, thanks",                   "intent": "gratitude"},
    # Mental health questions
    {"text": "I have been feeling very anxious and cannot sleep", "intent": "asking_mental_health_question"},
    {"text": "How can I cope with depression?",                 "intent": "asking_mental_health_question"},
    {"text": "I feel hopeless and empty all the time",          "intent": "asking_mental_health_question"},
    {"text": "My relationship is causing me a lot of stress",   "intent": "asking_mental_health_question"},
    {"text": "I have panic attacks frequently",                 "intent": "asking_mental_health_question"},
    {"text": "How do I deal with grief after losing someone?",  "intent": "asking_mental_health_question"},
    # Out of scope
    {"text": "What is the capital of France?",                  "intent": "out_of_scope"},
    {"text": "Write me a Python script to sort a list",        "intent": "out_of_scope"},
    {"text": "What is the recipe for pasta carbonara?",         "intent": "out_of_scope"},
    {"text": "Who won the World Cup in 2022?",                  "intent": "out_of_scope"},
]

# ── System Prompt ──────────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    intent_list = "\n".join(
        f"  - {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items()
    )
    few_shot = "\n".join(
        f'  Input: "{ex["text"]}" → intent: "{ex["intent"]}"'
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are an intent classification engine for a mental health support chatbot.

Your job is to classify the user's message into exactly ONE of these intents:
{intent_list}

Few-shot examples:
{few_shot}

RULES:
1. Reply ONLY with a valid JSON object — no markdown, no explanation outside JSON.
2. JSON format:
   {{
     "intent": "<one of the 5 intents>",
     "confidence": "<high|medium|low>",
     "reasoning": "<one sentence explaining your choice>"
   }}
3. When in doubt between mental health and out_of_scope, prefer asking_mental_health_question.
4. Greetings that also contain a mental health question → asking_mental_health_question.
"""


# ── Rule-based Fallback ────────────────────────────────────────────────────────
_GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
_GOODBYE_WORDS  = {"bye", "goodbye", "see you", "talk later", "farewell", "take care", "good night"}
_GRATITUDE_WORDS = {"thank", "thanks", "grateful", "appreciate", "cheers", "thx", "ty"}
_MH_KEYWORDS = {
    "anxious", "anxiety", "depressed", "depression", "sad", "hopeless", "stress",
    "stressed", "worried", "panic", "trauma", "grief", "lonely", "loneliness",
    "suicidal", "self-harm", "therapy", "therapist", "counseling", "mental health",
    "cope", "coping", "emotion", "feeling", "mood", "sleep", "insomnia", "worthless",
    "empty", "numb", "overwhelmed", "burnout", "phobia", "ocd", "ptsd",
}


def _rule_based_predict(text: str) -> Dict[str, Any]:
    """Simple keyword-based fallback when the LLM API is unavailable."""
    lower = text.lower().strip()
    words = set(re.findall(r"\w+", lower))

    if any(g in lower for g in _GOODBYE_WORDS):
        return {"intent": "goodbye", "confidence": "medium", "reasoning": "Farewell keywords detected."}
    if any(g in lower for g in _GRATITUDE_WORDS):
        return {"intent": "gratitude", "confidence": "medium", "reasoning": "Thank-you keywords detected."}
    if any(g in lower for g in _GREETING_WORDS):
        return {"intent": "greeting", "confidence": "medium", "reasoning": "Greeting keywords detected."}
    if words & _MH_KEYWORDS:
        return {
            "intent": "asking_mental_health_question",
            "confidence": "medium",
            "reasoning": "Mental health keywords detected.",
        }
    return {"intent": "out_of_scope", "confidence": "low", "reasoning": "No known pattern matched."}


# ── Classifier Class ───────────────────────────────────────────────────────────
class IntentClassifier:
    """
    Few-shot intent classifier backed by Groq LLM.
    Automatically falls back to rule-based classification on API errors.
    """

    def __init__(self, groq_api_key: str | None = None, model: str = "openai/gpt-oss-20b"):
        self._api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model
        self._client = None
        self._system_prompt = _build_system_prompt()
        self._init_client()

    def _init_client(self) -> None:
        if not self._api_key:
            logger.warning("No GROQ_API_KEY — intent classifier will use rule-based fallback.")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
            logger.info("Groq client initialized for intent classification.")
        except ImportError:
            logger.warning("groq package not installed — using rule-based fallback.")

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify the intent of *text*.

        Returns
        -------
        {
            "intent": "asking_mental_health_question",
            "confidence": "high",
            "reasoning": "User describes anxiety and sleep issues."
        }
        """
        if not text or not text.strip():
            return {"intent": "greeting", "confidence": "low", "reasoning": "Empty input."}

        if self._client is None:
            return _rule_based_predict(text)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system",  "content": self._system_prompt},
                    {"role": "user",    "content": text.strip()},
                ],
                temperature=0.0,   # deterministic for classification
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse(raw)

        except Exception as e:
            logger.error(f"Groq API error in intent classification: {e}")
            return _rule_based_predict(text)

    def _parse(self, raw: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM."""
        try:
            # Strip possible markdown code fences
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            data = json.loads(cleaned)
            intent = data.get("intent", "out_of_scope")
            if intent not in VALID_INTENTS:
                intent = "out_of_scope"
            return {
                "intent":     intent,
                "confidence": data.get("confidence", "medium"),
                "reasoning":  data.get("reasoning", ""),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse intent JSON: {raw!r}")
            return {"intent": "out_of_scope", "confidence": "low", "reasoning": "Parse error."}

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def needs_rag(self, text: str) -> bool:
        """Quick helper: True only if intent is 'asking_mental_health_question'."""
        return self.predict(text)["intent"] == "asking_mental_health_question"


# ── CLI entry-point ────────────────────────────────────────────────────────────
import os

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    clf = IntentClassifier(groq_api_key=os.getenv("GROQ_API_KEY"))
    samples = [
        "Hello there!",
        "Thank you so much for your support.",
        "Goodbye, see you tomorrow.",
        "I feel very depressed and cannot cope with life.",
        "What is the capital of France?",
        "I keep having panic attacks at night, what should I do?",
    ]
    print("\n=== Intent Classification Test ===")
    for s in samples:
        r = clf.predict(s)
        print(f"  [{r['intent']:35}] ({r['confidence']:6}) → {s}")
