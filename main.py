import threading
from picamera2 import Picamera2
import cv2
import torch
import telepot
import time
from datetime import datetime

# --- Telegram Bot Setup ---
bot_token = 'YOUR_BOT_TOKEN' 
chat_id = 'YOUR_CHAT_ID'
bot = telepot.Bot(bot_token)

# --- Load YOLOv5s Model (COCO classes) ---
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# --- Initialize PiCamera2 ---
print("[INFO] Starting PiCamera2...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))  # Lower resolution for smoothness
picam2.start()
time.sleep(2)  # Allow camera to warm up

# Global variables
frame_skip = 5  # Number of frames to skip for detection
frame_count = 0

# --- Detection Thread ---
def detect_animals():
    global frame_count
    while True:
        if frame_count % frame_skip == 0:
            # Capture image from PiCamera2
            frame = picam2.capture_array()

            # Run inference
            results = model(frame)

            # Parse results
            detected = results.pandas().xyxy[0]
            animals = detected[detected['name'].isin(['cat', 'dog', 'bird', 'cow', 'horse', 'sheep'])]

            # If any animals detected
            if len(animals) > 0:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"animal_detected_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                # Send to Telegram
                with open(filename, 'rb') as photo:
                    bot.sendPhoto(chat_id, photo, caption=f"🐾 Animal detected at {timestamp}!\nDetected: {', '.join(animals['name'].values)}")

                print(f"[ALERT] Animal Detected! Sent to Telegram: {animals['name'].values}")
            time.sleep(1)  # Add a delay to reduce detection frequency

        frame_count += 1
        time.sleep(0.1)  # Small delay for smoother video feed

# --- Display Thread ---
def display_feed():
    while True:
        frame = picam2.capture_array()
        # Display the camera feed (no blocking inference)
        cv2.imshow("Pi Camera Feed", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Start Threads ---
detection_thread = threading.Thread(target=detect_animals, daemon=True)
display_thread = threading.Thread(target=display_feed, daemon=True)

detection_thread.start()
display_thread.start()

# Wait for threads to finish (in this case, forever)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Program stopped by user.")
    cv2.destroyAllWindows()
    picam2.stop()
