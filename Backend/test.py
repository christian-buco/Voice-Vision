import cv2
import mediapipe as mp
import easyocr
import numpy as np
from matplotlib import pyplot as plt
# import pytesseract

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # Open webcam

reader = easyocr.Reader(['en'])
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)  # Mirror effect
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get finger tip and MCP joint positions
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                # middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                # ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                # pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Get the position of index tip
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                #region slightly in front of the fingertip
                roi_size = 150
                x1, y1 = max(0, x - roi_size // 2), max(0, y - roi_size - 20)
                x2, y2 = min(w, x + roi_size // 2), min(h, y - 20)

                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    # text = pytesseract.image_to_string(gray_roi, config='--psm 6')  # OCR
                    results = reader.readtext(gray_roi, detail=0)

                    if results:  # If text is detected, display it
                        detected_text = " ".join(results).strip()
                        cv2.putText(frame, f"Text: {detected_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Index finger pointing up
                # Check if the index finger is extended while others are folded
                # if (index_tip.y < index_mcp.y and  
                #     middle_tip.y > index_mcp.y and
                #     ring_tip.y > index_mcp.y and
                #     pinky_tip.y > index_mcp.y):
                #     cv2.putText(frame, "Pointing!", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow('Hand Pointing Detection & Text Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
