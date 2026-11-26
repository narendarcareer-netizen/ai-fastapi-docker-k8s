"""Sentiment classifier: TF-IDF + logistic regression, trained at startup.

Deliberately small and dependency-light so the service runs anywhere
(CPU-only, no model download). In production this module is the seam where
you swap in a real model artifact — a Hugging Face transformer, an ONNX
export, or a model pulled from a registry — without touching the API layer.
"""
from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_VERSION = "sentiment-tfidf-lr-v1"
LABELS = ["negative", "positive"]

# Embedded training set: short product-review style sentences.
# Enough for a demo-grade classifier; not a production dataset.
TRAINING_DATA = [
    ("I love this product, it works perfectly", "positive"),
    ("Absolutely fantastic, exceeded my expectations", "positive"),
    ("Great quality and fast delivery, very happy", "positive"),
    ("This is the best purchase I have made all year", "positive"),
    ("Wonderful experience, highly recommended", "positive"),
    ("Excellent customer service and a great product", "positive"),
    ("Really impressed, it does exactly what it promises", "positive"),
    ("Superb build quality, worth every penny", "positive"),
    ("I am delighted with how well this works", "positive"),
    ("Five stars, would buy again without hesitation", "positive"),
    ("Smooth setup and flawless performance", "positive"),
    ("Brilliant design, intuitive and reliable", "positive"),
    ("A joy to use every single day", "positive"),
    ("Top-notch quality, I recommend it to everyone", "positive"),
    ("Works like a charm, very satisfied", "positive"),
    ("Outstanding value for money", "positive"),
    ("Fast, reliable, and easy to use", "positive"),
    ("I could not be happier with this", "positive"),
    ("I hate this product, it is completely useless", "negative"),
    ("Terrible quality, broke after one day", "negative"),
    ("Worst purchase ever, do not buy this", "negative"),
    ("Awful experience, nothing works as advertised", "negative"),
    ("Disappointed and frustrated, total waste of money", "negative"),
    ("Very poor customer service and a faulty product", "negative"),
    ("It stopped working within a week, unacceptable", "negative"),
    ("Cheap materials, feels like it will fall apart", "negative"),
    ("Confusing setup and constant errors", "negative"),
    ("I regret buying this, it is a scam", "negative"),
    ("Slow, buggy, and unreliable", "negative"),
    ("Defective on arrival, had to return it", "negative"),
    ("One star, would never recommend to anyone", "negative"),
    ("Overpriced for what it does, very unhappy", "negative"),
    ("Crashed repeatedly and lost my work", "negative"),
    ("Misleading description, nothing like the photos", "negative"),
    ("Poorly made and impossible to use", "negative"),
    ("Customer support ignored me completely", "negative"),
]


class SentimentModel:
    """Thin wrapper so the API layer never imports sklearn directly."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self.train_seconds: float = 0.0
        self.train_samples: int = 0

    def load(self) -> None:
        texts = [t for t, _ in TRAINING_DATA]
        labels = [1 if label == "positive" else 0 for _, label in TRAINING_DATA]
        started = time.perf_counter()
        self._pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, C=10)),
            ]
        )
        self._pipeline.fit(texts, labels)
        self.train_seconds = time.perf_counter() - started
        self.train_samples = len(texts)

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str) -> tuple[str, float]:
        if self._pipeline is None:
            raise RuntimeError("model not loaded")
        probs = self._pipeline.predict_proba([text])[0]
        idx = int(probs.argmax())
        return LABELS[idx], float(probs[idx])

    def explain(self, text: str, top_k: int = 5) -> tuple[str, float, list[dict]]:
        """Predict, plus the tokens that pushed the score toward the label.

        Contribution = TF-IDF value x logistic-regression coefficient,
        sign-flipped when the predicted class is negative so a positive
        contribution always means "evidence for the returned label".
        """
        if self._pipeline is None:
            raise RuntimeError("model not loaded")
        tfidf = self._pipeline.named_steps["tfidf"]
        clf = self._pipeline.named_steps["clf"]
        features = tfidf.transform([text])
        probs = clf.predict_proba(features)[0]
        idx = int(probs.argmax())
        coef = clf.coef_[0]  # binary LR: one coefficient row, class 1 vs 0
        sign = 1.0 if idx == 1 else -1.0
        values = features.toarray()[0] * coef * sign
        names = tfidf.get_feature_names_out()
        order = values.argsort()[::-1][:top_k]
        tokens = [
            {"token": names[i], "contribution": round(float(values[i]), 4)}
            for i in order
            if values[i] != 0
        ]
        return LABELS[idx], float(probs[idx]), tokens
