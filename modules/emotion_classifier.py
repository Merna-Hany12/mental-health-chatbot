"""
Module 2 — Emotion Classifier
================================
Multi-class classifier using a fine-tuned Transformer (DistilBERT).
Trained on the GoEmotions / dair-ai/emotion dataset.

Emotion labels: sadness, joy, love, anger, fear, surprise

Integration with RAG pipeline:
    from modules.emotion_classifier import EmotionClassifier

    clf = EmotionClassifier()
    result = clf.predict("I feel so hopeless and lost")
    # → {"emotion": "sadness", "confidence": 0.94, "tone_hint": "empathetic_supportive", ...}

    rag.ask(
        question      = user_text,
        emotion_label = result["emotion"],
        emotion_tone  = result["tone_hint"],
    )

Usage:
    python emotion_classifier.py --train     # fine-tune + save
    python emotion_classifier.py             # run demo inference table
"""

from __future__ import annotations

import os
import json
import logging
import warnings
import numpy as np
import torch
from collections import Counter
from pathlib import Path
from typing import Dict, List

# ── Suppress noisy warnings ────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
torch.set_warn_always(False)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).parent.parent / "models" / "emotion_model"
RESULTS_DIR = Path("emotion_results")

# ── Hyper-parameters ───────────────────────────────────────────────────────────
CONFIG = {
    "base_model"     : "distilbert-base-uncased",
    "dataset_name"   : "dair-ai/emotion",
    "dataset_config" : "split",           # 16k train / 2k val / 2k test
    "max_length"     : 128,
    "batch_size"     : 32,
    "num_epochs"     : 5,
    "learning_rate"  : 2e-5,
    "weight_decay"   : 0.01,
    "warmup_ratio"   : 0.1,
    "seed"           : 42,
    "fp16"           : torch.cuda.is_available(),
}

# ── Label maps ─────────────────────────────────────────────────────────────────
LABEL2ID: Dict[str, int] = {
    "sadness" : 0,
    "joy"     : 1,
    "love"    : 2,
    "anger"   : 3,
    "fear"    : 4,
    "surprise": 5,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# ── Emotion → RAG tone hint ────────────────────────────────────────────────────
# Single source of truth — values are passed directly into RAGPipeline.ask()
# as emotion_tone=result["tone_hint"].
EMOTION_TONE: Dict[str, str] = {
    "sadness" : "empathetic_supportive",
    "joy"     : "celebratory_positive",
    "love"    : "warm_affirming",
    "anger"   : "calm_de-escalating",
    "fear"    : "reassuring_grounding",
    "surprise": "curious_engaging",
}


# ── Preprocessing ──────────────────────────────────────────────────────────────
class EmotionDataPreprocessor:
    """Load, clean, and tokenise the dair-ai/emotion dataset."""

    def __init__(self, tokenizer_name: str, max_length: int):
        from transformers import AutoTokenizer
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Light cleaning for Twitter-sourced text.
        Heavy normalisation (stemming, stopwords) is skipped intentionally —
        DistilBERT learns contextual representations directly from tokens.
        """
        text = text.strip()
        text = " ".join(text.split())   # collapse extra whitespace / newlines
        text = text.lower()
        return text

    def tokenize(self, examples: Dict) -> Dict:
        cleaned = [self.clean_text(t) for t in examples["text"]]
        return self.tokenizer(
            cleaned,
            padding    = "max_length",
            truncation = True,
            max_length = self.max_length,
        )

    def prepare_datasets(self, dataset_name: str, config: str):
        from datasets import load_dataset
        logger.info(f"Loading dataset: {dataset_name} / {config}")
        raw = load_dataset(dataset_name, config)

        logger.info("Tokenising ...")
        tokenized = raw.map(self.tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format("torch")

        logger.info(
            f"Sizes → train: {len(tokenized['train'])} | "
            f"val: {len(tokenized['validation'])} | "
            f"test: {len(tokenized['test'])}"
        )
        return tokenized


# ── Metrics ────────────────────────────────────────────────────────────────────
def build_compute_metrics():
    """Returns compute_metrics function used by HF Trainer during eval."""
    import evaluate
    from sklearn.metrics import f1_score

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions    = np.argmax(logits, axis=-1)
        acc = metric.compute(predictions=predictions, references=labels)
        f1  = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc["accuracy"], "f1_weighted": f1}

    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────
def train(save: bool = True) -> None:
    """
    Fine-tune DistilBERT on the dair-ai/emotion dataset
    (6-class: sadness, joy, love, anger, fear, surprise).
    Uses a WeightedTrainer to handle class imbalance.
    Saves HF model weights + tokenizer to MODEL_DIR.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from torch import nn
    except ImportError:
        raise ImportError(
            "Run: pip install transformers datasets evaluate scikit-learn accelerate torch"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CONFIG["seed"])

    # ── Preprocessing ──────────────────────────────────────────────────────────
    preprocessor = EmotionDataPreprocessor(
        tokenizer_name = CONFIG["base_model"],
        max_length     = CONFIG["max_length"],
    )
    datasets = preprocessor.prepare_datasets(
        dataset_name = CONFIG["dataset_name"],
        config       = CONFIG["dataset_config"],
    )

    # ── Class weights (handle imbalance) ──────────────────────────────────────
    train_labels = [int(datasets["train"][i]["labels"]) for i in range(len(datasets["train"]))]
    label_counts = Counter(train_labels)
    total        = len(train_labels)
    num_classes  = len(LABEL2ID)

    class_weights = torch.tensor(
        [total / (num_classes * label_counts[i]) for i in range(num_classes)],
        dtype=torch.float,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Class weights:")
    for i, w in enumerate(class_weights):
        logger.info(f"  {ID2LABEL[i]:<10} count={label_counts[i]:<5}  weight={w:.3f}")

    # ── Custom Trainer with weighted cross-entropy loss ────────────────────────
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss    = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ── Build model ────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"],
        num_labels = num_classes,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    )

    total_steps  = (len(datasets["train"]) // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir                  = str(MODEL_DIR),
        num_train_epochs            = CONFIG["num_epochs"],
        per_device_train_batch_size = CONFIG["batch_size"],
        per_device_eval_batch_size  = CONFIG["batch_size"],
        learning_rate               = CONFIG["learning_rate"],
        weight_decay                = CONFIG["weight_decay"],
        warmup_steps                = warmup_steps,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_weighted",
        greater_is_better           = True,
        fp16                        = CONFIG["fp16"],
        seed                        = CONFIG["seed"],
        logging_dir                 = str(RESULTS_DIR / "logs"),
        logging_steps               = 50,
        report_to                   = "none",
    )

    trainer = WeightedTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = datasets["train"],
        eval_dataset    = datasets["validation"],
        compute_metrics = build_compute_metrics(),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training ...")
    trainer.train()

    # ── Save best model + tokenizer ────────────────────────────────────────────
    if save:
        trainer.save_model(str(MODEL_DIR))
        preprocessor.tokenizer.save_pretrained(str(MODEL_DIR))
        logger.info(f"Model saved to: {MODEL_DIR}")

    # ── Evaluation on test set ─────────────────────────────────────────────────
    evaluate_model(trainer, datasets["test"])


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(trainer, test_dataset) -> None:
    """Run evaluation on test set, print metrics, save results and confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        f1_score,
    )

    logger.info("Evaluating on test set ...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_output = trainer.predict(test_dataset)
    logits      = predictions_output.predictions
    true_labels = predictions_output.label_ids
    pred_labels = np.argmax(logits, axis=-1)

    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    acc    = accuracy_score(true_labels, pred_labels)
    f1_w   = f1_score(true_labels, pred_labels, average="weighted")
    f1_mac = f1_score(true_labels, pred_labels, average="macro")
    report = classification_report(true_labels, pred_labels, target_names=label_names)
    cm     = confusion_matrix(true_labels, pred_labels)

    print(f"\n{'='*55}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (weighted) : {f1_w:.4f}")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print(f"{'='*55}")
    print("\nClassification Report:\n")
    print(report)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = label_names,
        yticklabels = label_names,
    )
    plt.title("Confusion Matrix — Emotion Classifier")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "confusion_matrix.png"), dpi=150)
    plt.close()

    # Persist metrics
    metrics = {
        "accuracy"        : round(acc, 4),
        "f1_weighted"     : round(f1_w, 4),
        "f1_macro"        : round(f1_mac, 4),
        "confusion_matrix": cm.tolist(),
    }
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    logger.info("Results saved to emotion_results/")


# ── Inference Class ────────────────────────────────────────────────────────────
class EmotionClassifier:
    """
    Production inference wrapper.
    Load once, call predict() on every user message in the chatbot pipeline.

    predict() returns emotion + confidence + tone_hint ready to be passed
    directly into RAGPipeline.ask():

        result = clf.predict(user_text)
        rag.ask(
            question      = user_text,
            emotion_label = result["emotion"],
            emotion_tone  = result["tone_hint"],
        )
        if result["crisis_flag"]:
            trigger_crisis_escalation()
    """

    def __init__(self, model_dir: str = str(MODEL_DIR)):
        from transformers import pipeline as hf_pipeline, AutoTokenizer
        logger.info(f"Loading emotion model from '{model_dir}' ...")

        # Load tokenizer explicitly and disable token_type_ids —
        # DistilBERT does not use them but some tokenizer configs pass them anyway.
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

        self._pipe = hf_pipeline(
            task       = "text-classification",
            model      = model_dir,
            tokenizer  = tokenizer,
            top_k      = None,            # return scores for all 6 classes
            truncation = True,
            max_length = CONFIG["max_length"],
            device     = 0 if torch.cuda.is_available() else -1,
        )
        logger.info("Emotion classifier ready")

    def predict(self, text: str) -> Dict:
        """
        Classify the dominant emotion in *text*.

        Returns a dict ready to unpack into RAGPipeline.ask():
        {
          "emotion"    : "sadness",                → emotion_label in RAG prompt
          "confidence" : 0.94,
          "tone_hint"  : "empathetic_supportive",  → emotion_tone  in RAG prompt
          "all_scores" : {"sadness": 0.94, ...},
          "crisis_flag": False
        }
        """
        if not text or not text.strip():
            return {
                "emotion"    : "sadness",
                "confidence" : 0.0,
                "tone_hint"  : EMOTION_TONE["sadness"],
                "all_scores" : {},
                "crisis_flag": bool(False),
            }

        cleaned = EmotionDataPreprocessor.clean_text(text)
        scores  = self._pipe(cleaned)[0]
        top     = max(scores, key=lambda x: x["score"])
        emotion = top["label"]

        return {
            "emotion"    : emotion,
            "confidence" : round(top["score"], 4),
            "tone_hint"  : EMOTION_TONE.get(emotion, "neutral"),
            "all_scores" : {s["label"]: round(s["score"], 4) for s in scores},
            "crisis_flag": bool(emotion in {"sadness", "fear"} and top["score"] >= 0.90),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict emotions for a list of texts in one pass."""
        cleaned = [EmotionDataPreprocessor.clean_text(t) for t in texts]
        batch   = self._pipe(cleaned)
        results = []
        for scores in batch:
            top     = max(scores, key=lambda x: x["score"])
            emotion = top["label"]
            results.append({
                "emotion"    : emotion,
                "confidence" : round(top["score"], 4),
                "tone_hint"  : EMOTION_TONE.get(emotion, "neutral"),
                "all_scores" : {s["label"]: round(s["score"], 4) for s in scores},
                "crisis_flag": emotion in {"sadness", "fear"} and top["score"] >= 0.90,
            })
        return results


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train(save=True)
    else:
        clf = EmotionClassifier()

        test_sentences = [
            "I feel so hopeless and empty inside.",
            "Today was absolutely wonderful, I am over the moon!",
            "I am furious, this is completely unacceptable.",
            "Something feels off, I am a bit scared.",
            "I love spending time with my family so much.",
            "Wow, I did not see that coming at all!",
            "Everything is going great, I feel fantastic!",
            "Oh my goodness, I am completely speechless!",
            "I am so incredibly blessed to have you in my life.",
        ]

        print(f"\n{'Text':<52} {'Emotion':<10} {'Conf':>6}  {'Tone Hint':<25} {'Crisis'}")
        print("-" * 110)
        for text in test_sentences:
            r = clf.predict(text)
            print(
                f"{text[:50]:<52} "
                f"{r['emotion']:<10} "
                f"{r['confidence']:>6.2f}  "
                f"{r['tone_hint']:<25} "
                f"{'⚠️ YES' if r['crisis_flag'] else 'no'}"
            )