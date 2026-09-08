

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY_ENV_VAR = "GOOGLE_API_KEY"
MODEL_NAME = "gemini-3.5-flash"

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV_VAR} was not found. Add it to your .env file "
            "before calling generate_precaution() — never hardcode API keys in code."
        )
    _client = genai.Client(api_key=api_key)
    return _client


def generate_precaution(predicted_class: str, confidences: dict, base_precaution: str) -> str:
    """
    Ask Gemini to turn the rule-based precaution into a short, natural,
    situation-specific message — grounded entirely in the facts passed in.

    Raises on any failure (network, auth, rate limit) — callers should
    catch this and fall back to base_precaution, since the tool should
    still work if Gemini is unavailable.
    """
    client = _get_client()

    conf_summary = ", ".join(f"{k}: {v * 100:.0f}%" for k, v in confidences.items())

    prompt = f"""You are a plant care assistant helping a home gardener understand
a leaf scan result. Do not diagnose independently — only use the facts below.

Predicted condition: {predicted_class}
Confidence breakdown: {conf_summary}
Standard guidance for this condition: {base_precaution}

Write a short, friendly precaution message (3-4 sentences) for the plant
owner based ONLY on the information above.
- Do not change the diagnosis or suggest a different condition.
- Do not invent confidence numbers beyond what's given above.
- If confidence is not high, or another condition shows a meaningful
  score, gently suggest a follow-up check or a clearer photo.
- Keep it practical and specific to this condition, not generic.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")

    return text