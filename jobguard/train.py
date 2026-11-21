"""Train and compare models, then save the best one.

Usage:
    python -m jobguard.train --data data/fake_job_postings.csv

Only about 5% of postings are fake, so accuracy is misleading (predicting
"real" every time already gives 95%). We compare models on the fake class:
precision, recall, F1 and PR-AUC.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import FLAG_FEATURES, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parent.parent


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=60000,
                           sublinear_tf=True)


def make_models():
    text_only = ColumnTransformer([("text", tfidf(), "text")])
    text_and_flags = lambda: ColumnTransformer([("text", tfidf(), "text"),  # noqa: E731
                                                ("flags", StandardScaler(), FLAG_FEATURES)])
    return {
        "Naive Bayes (text)": Pipeline([("features", text_only), ("model", ComplementNB(alpha=0.3))]),
        "Logistic Regression (text + flags)": Pipeline([
            ("features", text_and_flags()),
            ("model", LogisticRegression(C=8, class_weight="balanced", max_iter=3000))]),
        "Linear SVM (text + flags)": Pipeline([
            ("features", text_and_flags()),
            ("model", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3))]),
    }


def best_threshold(y, prob):
    """Pick the cut-off that gives the best F1 on out-of-fold predictions."""
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {"precision": round(precision_score(y, pred), 3), "recall": round(recall_score(y, pred), 3),
            "f1": round(f1_score(y, pred), 3), "pr_auc": round(average_precision_score(y, prob), 3),
            "roc_auc": round(roc_auc_score(y, prob), 3), "threshold": round(threshold, 3)}


def signal_rates(X, y):
    """How often postings with each warning sign were fake. Shown in the app."""
    checks = {"has_company_logo": 0, "no_company_profile": 1, "has_questions": 0,
              "no_requirements": 1, "telecommuting": 1}
    out = {}
    for col, bad in checks.items():
        mask = X[col] == bad
        out[col] = {"warning_value": bad, "fake_rate_with": round(float(y[mask].mean()), 3),
                    "fake_rate_without": round(float(y[~mask].mean()), 3), "posts": int(mask.sum())}
    return out


def save_plots(y_test, prob, threshold, name):
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    p, r, _ = precision_recall_curve(y_test, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, color="#2563eb", lw=2)
    ax.set_xlabel("Recall (fake posts caught)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve\n{name}")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(reports / "pr_curve.png", dpi=130)
    plt.close(fig)

    cm = confusion_matrix(y_test, (prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "#111", fontsize=13)
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "fake_job_postings.csv"))
    args = ap.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"{len(X):,} postings, {int(y.sum())} fake ({y.mean():.1%}). Train {len(X_train):,} / test {len(X_test):,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in make_models().items():
        # threshold is chosen on the training data only (out-of-fold), never on the test set
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y_train, oof)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = scores(y_test, prob, thr)
        fitted[name] = (model, thr, prob)
        print(f"  {name:<36} F1 {results[name]['f1']:.3f}  recall {results[name]['recall']:.3f}  "
              f"precision {results[name]['precision']:.3f}  PR-AUC {results[name]['pr_auc']:.3f}")

    best = max(results, key=lambda k: results[k]["f1"])
    model, thr, prob = fitted[best]
    save_plots(y_test, prob, thr, best)

    # refit the chosen model on all data for the app (same settings, same threshold)
    final = make_models()[best]
    final.fit(X, y)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final, models_dir / "model.joblib", compress=3)

    lstm_file = ROOT / "reports" / "lstm_metrics.json"
    if lstm_file.exists():
        results["BiLSTM (text, PyTorch)"] = json.loads(lstm_file.read_text())

    test_idx = X_test.index
    examples = pd.read_csv(args.data).loc[test_idx].assign(prob=prob)
    picks = pd.concat([examples[examples.fraudulent == 1].nlargest(3, "prob"),
                       examples[examples.fraudulent == 0].sample(3, random_state=SEED)])
    keep = ["title", "location", "company_profile", "description", "requirements", "benefits",
            "employment_type", "salary_range", "has_company_logo", "has_questions", "telecommuting", "fraudulent"]
    (models_dir / "examples.json").write_text(picks[keep].fillna("").to_json(orient="records", indent=1))

    info = {"chosen_model": best, "threshold": thr, "test_results": results,
            "rows": int(len(X)), "fake_share": round(float(y.mean()), 4),
            "signals": signal_rates(X, y)}
    (models_dir / "metrics.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved {best} to models/model.joblib (threshold {thr:.3f})")


if __name__ == "__main__":
    main()
