import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Define paths to your folders
happy_folder = "dataset/happy"
sad_folder = "dataset/sad"

# Initialize lists for images and labels
images = []
labels = []

# Load happy images
for filename in os.listdir(happy_folder):
    img_path = os.path.join(happy_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    if img is not None:
        img = cv2.resize(img, (48, 48))  # Resize to 48x48
        images.append(img)
        labels.append(0)  # Label for happy

# Load sad images
for filename in os.listdir(sad_folder):
    img_path = os.path.join(sad_folder, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (48, 48))
        images.append(img)
        labels.append(1)  # Label for sad

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values
images = images.astype("float32") / 255.0
images = np.expand_dims(images, axis=-1)  # Add channel dimension (N, 48, 48, 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)


# Define the CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),  # 2 classes: happy, sad
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
)

# Save the model
model.save("emotion_model.h5")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_accuracy:.4f}")
