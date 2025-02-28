import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(2)  # Change Camera input. Austin uses 2 for his camera. Use 0 or something

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:       # results is the joint that stores objects in the frame
        frame = r.plot()    # r has the detection info for the entire frame
        detected_objects = []
        for box in r.boxes:
            class_id = int(box.cls[0])  
            confidence = box.conf[0]   
            object_name = model.names[class_id]  
            detected_objects.append(object_name) 

            print(f"Detected: {object_name} ({confidence:.2f})")    # Printing this joint to see whats going on

        print(f"Detected objects: {', '.join(detected_objects)}")

    cv2.imshow("AI Glasses Feed", frame)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()