"""FastAPI app. Run locally with:  uvicorn jobguard.api:app --reload"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predict import Detector

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="JobGuard", description="Detects fake job postings.", version="1.0")
detector = Detector()


class Posting(BaseModel):
    title: str = Field("", max_length=300)
    location: str = Field("", max_length=300)
    company_profile: str = Field("", max_length=20000)
    description: str = Field(..., min_length=20, max_length=30000)
    requirements: str = Field("", max_length=20000)
    benefits: str = Field("", max_length=20000)
    employment_type: str = Field("", max_length=50)
    salary_range: str = Field("", max_length=50)
    has_company_logo: bool = False
    has_questions: bool = False
    telecommuting: bool = False


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(posting: Posting):
    data = posting.model_dump()
    for key in ("has_company_logo", "has_questions", "telecommuting"):
        data[key] = int(data[key])
    return detector.predict(data)


@app.get("/api/examples")
def examples():
    return detector.examples


@app.get("/api/model")
def model_info():
    info = detector.info
    return {"chosen_model": info["chosen_model"], "rows": info["rows"], "fake_share": info["fake_share"],
            "test_results": info["test_results"]}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
