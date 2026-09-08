import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Load the saved model
model = tf.keras.models.load_model("plant_disease_model.keras")

# 2. Define your class labels (make sure they match your training folder order)
class_names = [
    "Healthy",
    "Early Blight",
    "Late Blight",
]  # Replace with your actual class names

# 3. Load and preprocess a new image
img_path = "test_image.jpg"  # Path to the image you want to classify
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# EfficientNet expects a batch of images, so add a batch dimension: (1, 224, 224, 3)
img_batch = np.expand_dims(img_array, axis=0)

# 4. Run prediction
predictions = model.predict(img_batch)
predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index] * 100

# 5. Output the result
print(f"Prediction: {class_names[predicted_index]}")
print(f"Confidence: {confidence:.2f}%")