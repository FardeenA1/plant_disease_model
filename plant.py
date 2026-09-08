import tensorflow as tf

# Standard image size and batch size for training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define the directory paths to your dataset folders
train_dir = "C:/Users/NADEEM ANSARI/Desktop/plant/Train"
val_dir = "C:/Users/NADEEM ANSARI/Desktop/plant/Validation"
test_dir = "C:/Users/NADEEM ANSARI/Desktop/plant/Test"
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

# Load Validation Data
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Load Test Data (Targeting the nested 'Test/Test' folder from your image)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}") 
# Should output: ['Healthy', 'Powdery', 'Rust']



from tensorflow.keras import layers, models

# Define the model architecture
model = models.Sequential([
    # Input Layer & Rescaling: Standardize pixel values from [0, 255] to [0, 1]
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    
    # Block 1: Find basic features (edges, corners)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 2: Find more complex features (textures, patterns)
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 3: Find high-level features specific to the plant diseases
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the 2D feature maps into a 1D vector
    layers.Flatten(),
    
    # Fully connected network to interpret the features
    layers.Dense(64, activation='relu'),
    
    # Output Layer: 3 neurons (for your 3 classes) using softmax to output percentages/probabilities
    layers.Dense(3, activation='softmax')
])

# View a summary of your network
model.summary()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.callbacks import ModelCheckpoint

# Set image shape matching your setup
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = len(class_names)  # Assuming class_names is defined above


# -------------------------------------------------------------------
# 1. Build EfficientNetB0
# -------------------------------------------------------------------
def build_efficientnetb0():
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained weights

    inputs = layers.Input(shape=IMG_SHAPE)
    # EfficientNetB0 handles standard [0, 255] inputs internally
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="EfficientNetB0")


# -------------------------------------------------------------------
# 2. Training Function with Auto-Save (ModelCheckpoint)
# -------------------------------------------------------------------
def train_transfer_model(
    model,
    train_data,
    val_data,
    epochs=10,
    save_filename="plant_disease_model.keras",
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Save the best model automatically based on validation loss
    checkpoint = ModelCheckpoint(
        filepath=save_filename, monitor="val_loss", save_best_only=True, verbose=1
    )

    print(f"\n================ Training {model.name} ================")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint],  # Automatically saves model during training
    )

    return history


# -------------------------------------------------------------------
# 3. Instantiate, Train, and Save
# -------------------------------------------------------------------
efficientnet_model = build_efficientnetb0()

history_efficientnet = train_transfer_model(
    model=efficientnet_model,
    train_data=train_dataset,
    val_data=val_dataset,
    epochs=10,
    save_filename="plant_disease_model.keras",
)

# Optional explicit save at the very end
efficientnet_model.save("plant_disease_model_final.keras")
print("Model successfully saved as plant_disease_model.keras")