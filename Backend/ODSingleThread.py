import cv2
import pyttsx3
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to queue speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Find Camera
cap = None
for i in range(10):  # Test indexes 0-9
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        print(f"Camera found at index {i}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None or not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Tracking objects
announced_objects = set()  # Objects already announced
last_seen = {}  # Last seen time of objects
FORGET_TIME = 3  # Forget objects after 3 seconds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence required to announce an object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = model(frame)

    detected_objects = set()  # Tracks objects in the current frame
    current_time = time.time()

    for r in results:
        frame = r.plot()  # Draw bounding boxes
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])  

            # If unsure, don't say
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
cap.release()
cv2.destroyAllWindows()
