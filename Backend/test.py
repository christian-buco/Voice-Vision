import cv2
import mediapipe as mp
import easyocr
import numpy as np
from matplotlib import pyplot as plt
import threading
import time
# import pytesseract

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # Open webcam
reader = easyocr.Reader(['en'])

detected_text = ""
txt_lock = threading.Lock()

def process_txt(roi):
    global detected_text
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray_roi, detail=0)
    with txt_lock:
        detected_text = " ".join(results).strip() if results else ""

with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    prev_time = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        text_detected = False
        roi = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get finger tip and MCP joint positions
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Get the position of index tip
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                z = index_tip.z

                #region slightly in front of the fingertip
                roi_size = 200
                x1, y1 = max(0, x - roi_size // 2), max(0, y - roi_size - 20)
                x2, y2 = min(w, x + roi_size // 2), min(h, y - 20)

                roi = frame[y1:y2, x1:x2]

                if (roi.size > 0 and index_tip.y < index_mcp.y and  
                    middle_tip.y > index_mcp.y and
                    ring_tip.y > index_mcp.y and
                    pinky_tip.y > index_mcp.y):
                    text_detected = True
                    threading.Thread(target=process_txt, args=(roi,), daemon=True).start()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

                # Index finger pointing up
                # Check if the index finger is extended while others are folded
                # if (index_tip.y < index_mcp.y and  
                #     middle_tip.y > index_mcp.y and
                #     ring_tip.y > index_mcp.y and
                #     pinky_tip.y > index_mcp.y):
                
                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        with txt_lock:
            if detected_text:
                cv2.putText(frame, f"Text: {detected_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif not text_detected:
                detected_text = ""

        end_time = time.time()
        fps = 1/(end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Hand Pointing Detection & Text Recognition', frame)

        # if roi is not None and roi.size > 0:
        #     cv2.imshow('ROI', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
