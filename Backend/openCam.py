import cv2
import pyttsx3
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()


for i in range(10):  # Test indexes 0-9
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        print(f"Camera found at index {i}")
        cap = temp_cap
        break
    temp_cap.release()  

# cap = cv2.VideoCapture(2)   # Change Camera input. Austin uses 2 for his camera. Use 0 or something


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

    for r in results:       # results is the joint that stores objects in the frame
        frame = r.plot()    # r has the detection info for the entire frame

        for box in r.boxes:
            class_id = int(box.cls[0])  ## class Id
            confidence = box.conf[0]   
            object_name = model.names[class_id]  
            
            # Get bounding box center
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2 
            box_width = box.xyxy[0][2] - box.xyxy[0][0]

            if box_width > width * 0.5:
                distance = "very close"
            if box_width < width * 0.3:
                distance = "near"
            else:
                distance = "far"

            # Determine the direction
            if x_center < left_region:
                direction = "on the left"
            elif x_center > right_region:
                direction = "on the right"
            else:
                direction = "in the center"

            detected_objects.add(object_name)
            object_directions.append(f"{object_name} {direction} {distance}")

           # print(f"Detected: {object_name} ({confidence:.2f}) {direction}")  # Debug print

    # Bringing back new_objects so only new objects can be announced 
    new_objects = detected_objects - announced_objects

    # Announce detected objects
    if new_objects:
        announcement = "I see " + ", ".join([f"{obj}" for obj in new_objects])
        print(announcement)  # Debug print
        engine.say(announcement)
        engine.runAndWait()  # Speak out loud
        announced_objects.update(new_objects)   # Store that the new object was announced
    else:
        print("No objects detected.")
        
    announced_objects.intersection_update(detected_objects)

    # print(f"Detected: {object_name} ({confidence:.2f})")    # Printing this joint to see whats going on

    print(f"Detected objects: {', '.join(detected_objects)}")

    cv2.imshow("AI Glasses Feed", frame)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()