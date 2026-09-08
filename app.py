import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_condition
from gemini_advisor import generate_precaution

# Set to False to skip Gemini entirely and always use the rule-based
# precaution text from predict.py (useful for offline testing/demos,
# or if you don't want the extra API latency/cost during a live viva).
USE_GEMINI = True

app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a plant leaf image and get back its condition, "
    "confidence scores, and recommended precautions.",
    version="1.0.0",
)

# Allow requests from your frontend during development.
# Tighten allow_origins to your actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.get("/")
def root():
    return {"status": "ok", "message": "Plant Disease Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns:
    {
        "predicted_class": "Rust",
        "confidences": {"Healthy": 0.03, "Powdery": 0.12, "Rust": 0.85},
        "precaution": "..."
    }
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save the upload to a temp file since predict_condition() expects a path.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    finally:
        file.file.close()

    try:
        result = predict_condition(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # clean up temp file

    if USE_GEMINI:
        try:
            enriched = generate_precaution(
                result["predicted_class"], result["confidences"], result["precaution"]
            )
            result["precaution"] = enriched
            result["precaution_source"] = "gemini"
        except Exception as e:
            # Gemini failed (no key, rate limit, network issue) — fall back
            # to the rule-based precaution so the app still works.
            print(f"[gemini_advisor] falling back to rule-based precaution: {e}")
            result["precaution_source"] = "rule-based (gemini unavailable)"
    else:
        result["precaution_source"] = "rule-based"

    return result