"""Loading and cleaning the job postings.

The model sees two kinds of input:
1. Text: title, company profile, description, requirements, benefits and a few
   short fields, joined into one string.
2. Simple flags: does the post have a company logo, screening questions,
   a company profile, a salary, and how long is the text.
"""
import re

import numpy as np
import pandas as pd

TEXT_FIELDS = ["title", "location", "department", "company_profile", "description",
               "requirements", "benefits", "employment_type", "required_experience",
               "required_education", "industry", "function"]

FLAG_FEATURES = ["has_company_logo", "has_questions", "telecommuting",
                 "no_company_profile", "no_requirements", "no_benefits",
                 "has_salary", "log_text_length"]

URL_TOKEN = re.compile(r"#(url|email|phone)_[a-z0-9]+#", re.I)   # dataset masks these
HTML_TAG = re.compile(r"<[^>]+>")
NON_WORD = re.compile(r"[^a-z0-9$€£₹%+\s]")
SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text or "")
    text = URL_TOKEN.sub(lambda m: f" {m.group(1).lower()}token ", text)
    text = HTML_TAG.sub(" ", text)
    text = re.sub(r"&[a-z]+;", " ", text)                # &nbsp; &amp; ...
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # "teamWe" -> "team We" (lost line breaks)
    text = NON_WORD.sub(" ", text.lower())
    return SPACES.sub(" ", text).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Works for the full dataset and for a single posting typed into the app."""
    df = df.copy()
    for col in TEXT_FIELDS + ["salary_range"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in ["has_company_logo", "has_questions", "telecommuting"]:
        if col not in df:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    raw = df[TEXT_FIELDS].agg(" ".join, axis=1)
    out = pd.DataFrame({"text": raw.map(clean_text)}, index=df.index)
    out["has_company_logo"] = df["has_company_logo"]
    out["has_questions"] = df["has_questions"]
    out["telecommuting"] = df["telecommuting"]
    out["no_company_profile"] = (df["company_profile"].str.strip() == "").astype(int)
    out["no_requirements"] = (df["requirements"].str.strip() == "").astype(int)
    out["no_benefits"] = (df["benefits"].str.strip() == "").astype(int)
    out["has_salary"] = (df["salary_range"].str.strip() != "").astype(int)
    out["log_text_length"] = np.log1p(out["text"].str.len())
    return out


def load_dataset(path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return build_features(df), df["fraudulent"].astype(int)
