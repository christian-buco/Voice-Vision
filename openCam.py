import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # Trying to open the camera with index 1 (may differ depending on your system)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results: 
        frame = r.plot()

    cv2.imshow("AI Glasses Feed", frame)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()