"""
train_models.py — Train & save Modules 1 and 2
================================================
Run this once to train the language detector and emotion classifier
and save them to the saved_models/ directory.

Module 3 (Intent) uses the Groq LLM API — no training needed.
Module 4 (RAG) uses a pre-built index in Qdrant — run index_qdrant.py.

Usage:
    python train_models.py [--module 1] [--module 2] [--all]
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

Path("saved_models").mkdir(exist_ok=True)


def train_language_detector():
    print("\n" + "="*60)
    print("  Module 1 — Language Detector Training")
    print("="*60)
    from modules.language_detector import train
    pipeline = train(save=True)
    print("✅  Language detector saved to saved_models/language_detector.joblib")
    return pipeline


def train_emotion_classifier():
    print("\n" + "="*60)
    print("  Module 2 — Emotion Classifier Training")
    print("="*60)
    from modules.emotion_classifier import train
    train(save=True)
    print("✅  Emotion classifier saved to saved_models/emotion_classifier/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chatbot modules")
    parser.add_argument("--module", type=int, choices=[1, 2], action="append",
                        help="Which module to train (1=language, 2=emotion)")
    parser.add_argument("--all", action="store_true", help="Train all modules")
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.print_help()
        sys.exit(0)

    modules_to_train = [1, 2] if args.all else (args.module or [])

    if 1 in modules_to_train:
        train_language_detector()
    if 2 in modules_to_train:
        train_emotion_classifier()

    print("\n🎉  All requested models trained successfully!")
    print("    Next steps:")
    print("    1. Run index_qdrant.py to index the mental health dataset")
    print("    2. Run uvicorn main:app --reload to start the server")
