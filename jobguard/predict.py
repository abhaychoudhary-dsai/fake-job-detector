"""Load the saved model and score one job posting, with a short explanation."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import FLAG_FEATURES, build_features

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SIGNAL_TEXT = {
    "has_company_logo": "No company logo",
    "no_company_profile": "No company profile",
    "has_questions": "No screening questions",
    "no_requirements": "No requirements listed",
    "telecommuting": "Work-from-home role",
}


class Detector:
    def __init__(self, models_dir: Path = MODELS):
        self.model = joblib.load(models_dir / "model.joblib")
        self.info = json.loads((models_dir / "metrics.json").read_text())
        self.examples = json.loads((models_dir / "examples.json").read_text())
        self.threshold = float(self.info["threshold"])
        features = self.model.named_steps["features"]
        self.feature_names = np.array([n.split("__", 1)[1] for n in features.get_feature_names_out()])
        self.coef = self._coefficients(self.model.named_steps["model"])

    @staticmethod
    def _coefficients(clf):
        if hasattr(clf, "coef_"):                       # logistic regression
            return np.asarray(clf.coef_).ravel()
        if hasattr(clf, "calibrated_classifiers_"):     # calibrated linear SVM
            return np.mean([c.estimator.coef_.ravel() for c in clf.calibrated_classifiers_], axis=0)
        return None

    def _explain(self, X, top=6):
        if self.coef is None:
            return [], []
        x = self.model.named_steps["features"].transform(X)
        x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
        contrib = x * self.coef
        is_word = ~np.isin(self.feature_names, FLAG_FEATURES)
        idx = np.where(is_word & (contrib != 0))[0]
        order = idx[np.argsort(contrib[idx])]
        fake = [{"term": str(self.feature_names[i]), "weight": round(float(contrib[i]), 3)} for i in order[::-1][:top] if contrib[i] > 0]
        real = [{"term": str(self.feature_names[i]), "weight": round(float(contrib[i]), 3)} for i in order[:top] if contrib[i] < 0]
        return fake, real

    def _signals(self, X):
        out = []
        for col, s in self.info["signals"].items():
            if int(X.iloc[0][col]) == s["warning_value"]:
                out.append({"signal": SIGNAL_TEXT[col], "fake_rate_with": s["fake_rate_with"],
                            "fake_rate_without": s["fake_rate_without"]})
        return out

    def predict(self, posting: dict) -> dict:
        X = build_features(pd.DataFrame([posting]))
        prob = float(self.model.predict_proba(X)[0, 1])
        fake_words, real_words = self._explain(X)
        if prob >= self.threshold:
            verdict = "Likely fake"
        elif prob >= self.threshold / 2:
            verdict = "Be careful"
        else:
            verdict = "Looks genuine"
        return {"fake_probability": round(prob, 4), "threshold": round(self.threshold, 3),
                "is_fake": prob >= self.threshold, "verdict": verdict,
                "words_pointing_fake": fake_words, "words_pointing_real": real_words,
                "warning_signs": self._signals(X)}
