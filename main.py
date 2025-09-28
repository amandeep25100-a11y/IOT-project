import cv2
import sqlite3
from ultralytics import YOLO
from datetime import datetime

# ---------------------------
# Load YOLOv8 model
# ---------------------------
model = YOLO("yolov8n.pt")

# ---------------------------
# Setup SQLite database
# ---------------------------
conn = sqlite3.connect("events.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   label TEXT,
                   confidence REAL,
                   x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                   dangerous INTEGER,
                   timestamp TEXT)''')
conn.commit()

# ---------------------------
# Define dangerous objects
# ---------------------------
dangerous_items = {"knife", "scissors", "gun", "sword"}

# ---------------------------
# Open webcam
# ---------------------------
cap = cv2.VideoCapture(0)

print("Press Q to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]   # class label
            conf = float(box.conf)            # confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

            # Check if object is dangerous
            is_dangerous = 1 if cls in dangerous_items else 0

            # Draw bounding box
            color = (0, 0, 255) if is_dangerous else (0, 255, 0)  # red = danger, green = safe
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save detection to DB
            cursor.execute('''INSERT INTO detections 
                              (label, confidence, x1, y1, x2, y2, dangerous, timestamp) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                           (cls, conf, x1, y1, x2, y2, is_dangerous,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    # Show the frame
    cv2.imshow("YOLO Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()
conn.close()
