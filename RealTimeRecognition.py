import cv2
import numpy as np
import tensorflow as tf
import os
from keras.applications.resnet import preprocess_input

# ---------------- SETTINGS ----------------
IMG_SIZE = 96
top, right, bottom, left = 100, 150, 400, 450

# Ensure these paths match your local setup
MODEL_PATH = r"model\ASL_ResNet50_Final.h5"
DATASET_PATH = r"Random3\train" 

# -----------------------------------------

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Get labels from the folder structure
labels = sorted(os.listdir(DATASET_PATH))
print("Loaded labels:", labels)

# CAP_DSHOW helps avoid startup delays on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ERROR: Webcam not detected.")
    exit()

print("🎥 Webcam started. Press 'Q' to exit.")

# -----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 1. Define ROI (Region of Interest)
    roi = frame[top:bottom, right:left]
    
    # 2. Preprocess ROI for ResNet50
    # ResNet50 expects specific color channel subtraction via preprocess_input
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype(np.float32))

    # 3. Prediction
    prediction = model.predict(img, verbose=0)
    char_index = np.argmax(prediction)
    confidence = prediction[0][char_index]

    # 4. Visualization
    predicted_char = labels[char_index]
    
    # Only show prediction if confidence is above 50%
    if confidence > 0.50:
        display_text = f"{predicted_char} ({confidence*100:.1f}%)"
        color = (0, 255, 0) # Green for confident
    else:
        display_text = "Scanning..."
        color = (0, 0, 255) # Red for low confidence

    # Draw UI
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, display_text, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program Closed")