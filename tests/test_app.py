"""Tests run against the saved model in models/, so they do not need the dataset."""
from fastapi.testclient import TestClient

from jobguard.api import app, detector
from jobguard.data import build_features, clean_text

client = TestClient(app)


def test_clean_text_removes_html_and_masked_urls():
    assert clean_text("<p>Apply&nbsp;now at #URL_abc123#</p>") == "apply now at urltoken"


def test_features_flag_missing_company_profile():
    import pandas as pd
    X = build_features(pd.DataFrame([{"description": "We are hiring", "company_profile": ""}]))
    assert X.loc[0, "no_company_profile"] == 1


def test_known_examples_are_classified_correctly():
    for ex in detector.examples:
        result = detector.predict(ex)
        assert result["is_fake"] == bool(ex["fraudulent"]), ex["title"]


def test_predict_endpoint_returns_explanation():
    fake = next(e for e in detector.examples if e["fraudulent"])
    body = {k: fake[k] for k in ["title", "location", "company_profile", "description", "requirements",
                                 "benefits", "employment_type", "salary_range"]}
    body.update(has_company_logo=bool(fake["has_company_logo"]), has_questions=bool(fake["has_questions"]),
                telecommuting=bool(fake["telecommuting"]))
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["fake_probability"] <= 1
    assert data["words_pointing_fake"], "should list words that pushed the score up"


def test_short_description_is_rejected():
    assert client.post("/api/predict", json={"description": "hi"}).status_code == 422


def test_model_scores_are_published():
    info = client.get("/api/model").json()
    chosen = info["test_results"][info["chosen_model"]]
    assert chosen["f1"] > 0.85
