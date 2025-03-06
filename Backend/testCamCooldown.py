"""
This file is detects objects and says the object out loud.
- Threaded speaking 
- Puts announcments in queue
- Forgets an object if not in frame for 3 seconds to avoid repeating (can adjust)
- Add confidence threshold to say things its more sure of (can adjus)
"""
from picamera2 import Picamera2
import cv2
import pyttsx3
import threading
import queue
import time
from ultralytics import YOLO

try:
    model = YOLO("yolov8n.pt")
    print("Loaded Model")
except Exception as e:
    print(f"error loading model: {e}")
    exit()


# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
engine.setProperty('voice', 'english')
speech_queue = queue.Queue()

# Background speech worker thread
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Exit thread gracefully if None is received
        engine.say(text)
        engine.runAndWait()

# Start the speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Function to queue speech
def speak(text):
    speech_queue.put(text)

# Find Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Tracking objects
announced_objects = set()  # Objects already announced
last_seen = {}  # Last seen time of objects
FORGET_TIME = 3  # Forget objects after 3 seconds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence required to announce an object

while True:
    frame = picam2.capture_array()
    height, width, _ = frame.shape

    results = model(frame)

    detected_objects = set()  # Tracks objects in the current frame
    current_time = time.time()

    for r in results:
        frame = r.plot()  # Draw bounding boxes
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])  

            # If unsure, dont say
            if confidence < CONFIDENCE_THRESHOLD:
                continue  

            object_name = model.names[class_id]
            detected_objects.add(object_name)
            last_seen[object_name] = current_time  # Update last seen time

    # Forget objects that haven't been seen for a while
    for obj in list(last_seen.keys()):
        if current_time - last_seen[obj] > FORGET_TIME:
            last_seen.pop(obj, None)
            announced_objects.discard(obj)

    # Identify new objects
    new_objects = detected_objects - announced_objects

    if new_objects:
        announcement = "I see " + ", ".join(new_objects)
        print(announcement)
        speak(announcement)  # Announce only new objects
        announced_objects.update(new_objects)

    print(f"Currently detected objects: {', '.join(detected_objects)}")

    cv2.imshow("AI Glasses Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.close()
cv2.destroyAllWindows()
speech_queue.put(None)  
speech_thread.join()
