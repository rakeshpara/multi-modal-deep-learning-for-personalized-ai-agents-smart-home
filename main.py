import cv2
import numpy as np
import tensorflow as tf
import socket
import json
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

# BUILD MODEL
base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=outputs)

model.load_weights("model.weights.h5")
print("Model loaded successfully")

class_names = ['door_lock', 'door_unlock', 'light_off', 'light_on']

# Raspberry Pi IP
PI_IP   = "192.168.1.8"
PI_PORT = 5007


def send_gesture_to_pi(gesture, confidence):
    try:
        payload = json.dumps({
            "gesture":    gesture,
            "confidence": round(float(confidence), 4)
        })
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((PI_IP, PI_PORT))
        s.sendall(payload.encode())
        s.close()
        print(f"Sent to Pi: {gesture} ({confidence*100:.1f}%)")
    except Exception as e:
        print(f"Pi send error: {e}")


# Anker C200
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Could not open camera index 1. Trying index 0.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("No camera found.")
    exit()

print("Camera ready. Press q to exit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    box_size = 300
    cx, cy = w // 2, h // 2
    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    try:
        img = cv2.resize(roi, (224, 224))
    except:
        continue

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred       = model.predict(img, verbose=0)
    pred_index = np.argmax(pred)
    pred_class = class_names[pred_index]
    confidence = np.max(pred)

    # Send to Pi every 10 frames when confidence is high
    frame_count += 1
    if frame_count % 10 == 0 and confidence > 0.80:
        send_gesture_to_pi(pred_class, confidence)

    # Display prediction
    cv2.putText(frame, f"{pred_class}  {confidence*100:.1f}%", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if confidence > 0.80:
        actions = {
            "light_on":    "Light ON",
            "light_off":   "Light OFF",
            "door_lock":   "Door LOCK",
            "door_unlock": "Door UNLOCK"
        }
        action = actions[pred_class]
    else:
        action = "Detecting..."

    cv2.putText(frame, action, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

    # Status indicators
    cv2.putText(frame, f"Pi: {PI_IP}", (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(frame, "RL Agent: ACTIVE", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(frame, "Anker PowerConf C200", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Gesture Smart Home — RL Active", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()