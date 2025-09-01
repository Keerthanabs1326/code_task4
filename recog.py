import cv2
import numpy as np
import tensorflow as tf
from collections import deque

model = tf.keras.models.load_model(r"C:\app\gesture_model.h5")
class_names = {
    0: '01_palm', 1: '02_1', 2: '03_fist', 3: '04_fist_moved',
    4: '05_thumb', 5: '06_index', 6: '07_ok', 7: '08_palm_moved',
    8: '09_c', 9: '10_down'
}
gesture_actions = {
    "01_palm": "palm",
    "02_fist": "fist",
    "03_thumb": "thumb",
    "04_fist_moved": "fist moved",
    "05_thumb": "thumnb side",
    "06_index": "index",
    "07_ok": "ok",
    "08_palm_moved": "palm moved",
    "09_c": "side view",
    "10_down": "fingers"
}

print("Expected input shape:", model.input_shape)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

pred_buffer = deque(maxlen=5)

def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, 64, 64, 1)
    return reshaped

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    box_size = 200
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    input_tensor = preprocess(roi)
    preds = model.predict(input_tensor)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    pred_buffer.append(pred_class)
    smoothed_pred = max(set(pred_buffer), key=pred_buffer.count)

    action = gesture_actions.get(smoothed_pred, "Unknown Gesture")

    cv2.putText(frame, f"{smoothed_pred} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Action: {action}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    weighted_preds = {}
    for p in pred_buffer:
        weighted_preds[p] = weighted_preds.get(p, 0) + 1
        smoothed_pred = max(weighted_preds, key=weighted_preds.get)
    
    # Apply simple background subtraction
    bg_sub = cv2.createBackgroundSubtractorMOG2()
    mask = bg_sub.apply(roi)
    roi = cv2.bitwise_and(roi, roi, mask=mask)

    def predict_with_augmentation(roi):
        variants = [roi, cv2.flip(roi, 1)]
        preds = [model.predict(preprocess(v)) for v in variants]
        avg_pred = np.mean(preds, axis=0)
        return avg_pred
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    roi = cv2.bitwise_and(roi, roi, mask=mask)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()