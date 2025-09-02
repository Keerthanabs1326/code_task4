import cv2
import numpy as np
import tensorflow as tf
import os
import subprocess
from collections import deque

model = tf.keras.models.load_model(r"C:\app\gesture_model.h5")
class_names = {
    0: '01_palm', 1: '02_1', 2: '03_fist', 3: '04_fist_moved',
    4: '05_thumb', 5: '06_index', 6: '07_ok', 7: '08_palm_moved',
    8: '09_c', 9: '10_down'
}

gesture_actions = {
    "01_palm": lambda: print("🖐️ Palm detected — idle state"),
    "02_1": lambda: os.system("start calc"),  
    "03_fist": lambda: print("✊ Fist — maybe close window?"),
    "04_fist_moved": lambda: print("👊 Fist moved — action trigger"),
    "05_thumb": lambda: os.system("start notepad"),  
    "06_index": lambda: os.system("start mspaint"),  
    "07_ok": lambda: os.system("start ms-settings:"), 
    "08_palm_moved": lambda: os.system("start chrome"),  
    "09_c": lambda: subprocess.run(["powershell", "-Command", "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightness).CurrentBrightness"]), 
    "10_down": lambda: subprocess.run(["powershell", "-Command", "(Get-WmiObject -Class Win32_Battery).EstimatedChargeRemaining"]),  
}

print("Expected input shape:", model.input_shape)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

pred_buffer = deque(maxlen=5)
last_action = None

def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1)) 
    return reshaped

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    box_size = 200
    x1, y1 = w // 2 - box_size // 2, h// 2 - box_size // 2
    x2, y2 = x1 + box_size, y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    input_tensor = preprocess(roi)
    preds = model.predict(input_tensor, verbose=0)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    if confidence > 0.85:
        pred_buffer.append(pred_class)

    if pred_buffer:
        smoothed_pred = max(set(pred_buffer), key=pred_buffer.count)
    else:
        smoothed_pred = None


    action_fn = gesture_actions.get(smoothed_pred)

    if smoothed_pred != last_action and confidence > 0.85:
        if action_fn:
            action_fn()
        last_action = smoothed_pred

    cv2.putText(frame, f"{smoothed_pred} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Action: {smoothed_pred}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()