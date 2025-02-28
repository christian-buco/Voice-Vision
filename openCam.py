import cv2

cap = cv2.VideoCapture(0)  # Trying to open the camera with index 1 (may differ depending on your system)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("AI Glasses Feed", frame)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()