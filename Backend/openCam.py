import cv2
import pyttsx3
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak without blocking
def speak(text):
    thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
    thread.start()

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

announced_objects = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    left_region = width // 3
    right_region = 2 * (width // 3)

    results = model(frame)

    detected_objects = set()
    object_directions = []

    for r in results:
        frame = r.plot()

        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            object_name = model.names[class_id]

            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            box_width = box.xyxy[0][2] - box.xyxy[0][0]

            if box_width > width * 0.5:
                distance = "very close"
            elif box_width < width * 0.3:
                distance = "near"
            else:
                distance = "far"

            if x_center < left_region:
                direction = "on the left"
            elif x_center > right_region:
                direction = "on the right"
            else:
                direction = "in the center"

            detected_objects.add(object_name)
            object_directions.append(f"{object_name} {direction} {distance}")

    new_objects = detected_objects - announced_objects

    if new_objects:
        announcement = "I see " + ", ".join(new_objects)
        print(announcement)
        speak(announcement)  # Use the threaded speak function
        announced_objects.update(new_objects)
    else:
        print("No new objects detected.")

    announced_objects.intersection_update(detected_objects)

    print(f"Detected objects: {', '.join(detected_objects)}")

    cv2.imshow("AI Glasses Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
