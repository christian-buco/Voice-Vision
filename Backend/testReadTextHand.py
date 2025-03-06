"""
This file allows the user to point to text and have the text be read out loud.
- Detects text in area above index finger
- Thread for speech function
- Tracks last text spoken and puts cooldown
- Make OCR (text recognition) every couple of frames (improve perfomance)
- Adjust confidence level threshold, contrast, OCR call frequency to whatever you want
"""

from picamera2 import Picamera2
import cv2
import mediapipe as mp
import easyocr
import pyttsx3
import threading
import time
import queue

# Initialize EasyOCR, MediaPipe, and pyttsx3
reader = easyocr.Reader(['en'])
engine = pyttsx3.init(driverName='espeak')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Threaded speech function
def speak(text):
    thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True)
    thread.start()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Track last spoken text and cooldown
last_spoken_text = ""
last_spoken_time = 0
cooldown_seconds = 5  # Cooldown for last spoken. Adjust if needed

# Thread-safe queue for OCR processing
ocr_queue = queue.Queue()
ocr_thread_running = False  # Ensure only one OCR thread runs at a time

# OCR frequency control
ocr_frame_skip = 12  # Adjust if you want
frame_count = 0

# Function to process OCR asynchronously
def ocr_worker():
    global ocr_thread_running, last_spoken_text, last_spoken_time
    while True:
        roi = ocr_queue.get()  # Get ROI from the queue
        if roi is None:  # Stop thread if None is received
            break

        # Region of Interest (ROI). Text detection size
        resized_roi = cv2.resize(roi, (100, 50))  

        # Convert to grayscale and enhance contrast, helps with text recognition
        gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Refinements to OCR performance
        results = reader.readtext(
            thresh_roi,
            detail=0,
            contrast_ths=0.3,  # Lower contrast threshold
            text_threshold=0.2,  # Accept lower confidence text
            link_threshold=0.2   # More flexible word linking
        )

        if results:
            detected_text = " ".join(results).strip()
            current_time = time.time()

            # Speak only if new text is detected or cooldown is over
            if detected_text and (detected_text != last_spoken_text or current_time - last_spoken_time > cooldown_seconds):
                speak(detected_text)
                last_spoken_text = detected_text
                last_spoken_time = current_time

        ocr_thread_running = False  # Allow the next OCR request

# Start OCR processing thread
ocr_processing_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_processing_thread.start()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape
        
        # Process hand tracking
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Detection area near finger tips
                roi_size = 75  # Keep small size for speed
                x1, y1 = max(0, x - roi_size // 2), max(0, y - roi_size - 20)
                x2, y2 = min(w, x + roi_size // 2), min(h, y - 20)

                roi = frame[y1:y2, x1:x2]

                # Limit OCR calls to every certain number of frames (12 rn)
                if roi.size > 0 and not ocr_thread_running and frame_count % ocr_frame_skip == 0:
                    ocr_thread_running = True  # Mark OCR as running
                    ocr_queue.put(roi)  # Add ROI to the queue

                # Rectangle around text detection area (Only Visual)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw a circle at the index fingertip
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        frame_count += 1  # Increment frame counter
        cv2.imshow('Hand Pointing Text Reader', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
ocr_queue.put(None)  
ocr_processing_thread.join()

picam2.close()
cv2.destroyAllWindows()
