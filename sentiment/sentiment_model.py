"""
AI model layer: loads the local Hugging Face model and predicts sentiment.
Only this file knows about the model. Views and Reddit code just call predict().
"""
import logging

from django.conf import settings
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = settings.BASE_DIR / "ml_models" / "sentiment"
MAX_CHARS = 2000  # very long text is cut (model limit is 512 tokens)

_classifier = None  # loaded once, reused for every request


class ModelNotReadyError(Exception):
    """Raised when the model is not downloaded or fails to load."""


def _get_classifier():
    global _classifier
    if _classifier is None:
        if not MODEL_DIR.exists():
            raise ModelNotReadyError(
                "Model not found. Run: python manage.py download_model"
            )
        try:
            _classifier = pipeline(
                "sentiment-analysis",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
            )
            logger.info("Sentiment model loaded from %s", MODEL_DIR)
        except Exception as exc:
            raise ModelNotReadyError(f"Could not load model: {exc}") from exc
    return _classifier


def predict_many(texts):
    """Analyze a list of texts. Returns a list of dicts."""
    clean = [(t or "").strip()[:MAX_CHARS] for t in texts]
    valid = [t for t in clean if t]
    if not valid:
        return []

    results = _get_classifier()(valid, truncation=True, max_length=512)

    return [
        {
            "text": text,
            "sentiment": "Positive" if r["label"] == "POSITIVE" else "Negative",
            "confidence": round(float(r["score"]), 4),
        }
        for text, r in zip(valid, results)
    ]


def predict(text):
    """Analyze a single text. Returns one dict, or None if text is empty."""
    results = predict_many([text])
    return results[0] if results else None