

import sys
import numpy as np
import tensorflow as tf

# Must match training config in plant.py
IMG_SIZE = (224, 224)
MODEL_PATH = "plant_disease_model_final.keras"

# NOTE: class_names must match the order printed during training
# (tf.keras.utils.image_dataset_from_directory sorts folder names
# alphabetically, so this should be Healthy, Powdery, Rust — but
# double check against the "Classes found: [...]" print from plant.py
# before trusting this order).
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Precaution lookup — rule-based, not model-generated.
# These are general placeholders; swap in agronomy-sourced advice
# before this goes in front of real users.
PRECAUTIONS = {
    "Healthy": (
        "No signs of disease detected. Keep up regular watering and "
        "monitor leaves periodically for early signs of stress."
    ),
    "Powdery": (
        "Signs of powdery mildew detected. Improve air circulation around "
        "the plant, avoid overhead watering, and consider a sulfur-based "
        "or neem oil fungicide. Remove heavily affected leaves."
    ),
    "Rust": (
        "Signs of rust detected. Remove and destroy infected leaves, "
        "avoid wetting foliage when watering, and consider a copper-based "
        "or targeted rust fungicide. Isolate from other plants if possible."
    ),
}

_model = None  # loaded lazily, cached after first call


def load_model(model_path: str = MODEL_PATH):
    """Load and cache the trained model."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
    return _model


def preprocess_image(img_path: str) -> np.ndarray:
    """Load an image from disk and prepare it for the model."""
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def predict_condition(img_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Run inference on a single image.

    Returns a dict like:
    {
        "predicted_class": "Rust",
        "confidences": {"Healthy": 0.03, "Powdery": 0.12, "Rust": 0.85},
        "precaution": "..."
    }
    """
    model = load_model(model_path)
    img_array = preprocess_image(img_path)

    # Note: EfficientNetB0 handles [0, 255] inputs internally (as set up
    # in plant.py), so no manual rescaling here.
    predictions = model.predict(img_array, verbose=0)[0]

    confidences = {
        class_name: round(float(score), 4)
        for class_name, score in zip(CLASS_NAMES, predictions)
    }

    predicted_class = CLASS_NAMES[int(np.argmax(predictions))]

    return {
        "predicted_class": predicted_class,
        "confidences": confidences,
        "precaution": PRECAUTIONS[predicted_class],
    }


def _print_result(result: dict) -> None:
    print(f"\nPredicted condition: {result['predicted_class']}\n")
    print("Confidence breakdown:")
    for class_name, score in result["confidences"].items():
        print(f"  {class_name:10s}: {score * 100:.2f}%")
    print(f"\nPrecaution:\n  {result['precaution']}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_condition(image_path)
    _print_result(result)