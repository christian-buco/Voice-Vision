import cv2
import mediapipe as mp
import easyocr
import numpy as np
import pyttsx3
import threading
import time
import queue
from matplotlib import pyplot as plt
import speech_recognition as sr
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)  # Open webcam

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

reader = easyocr.Reader(['en'])
engine = pyttsx3.init()

engine.setProperty('rate', 150)

# Queue for speech announcements
speaking_queue = queue.Queue()

exit_flag = False

# Function to announce detected objects
def speak_text():
    global exit_flag
    while not exit_flag:
        try:
            text = speaking_queue.get(timeout=1)
            if text:
                print(f"🔊 Speaking: {text}")  # Debugging print
                engine.say(text)
                engine.runAndWait()
        except queue.Empty:
            pass


speech_thread = threading.Thread(target=speak_text, daemon=True)
speech_thread.start()

# Print available microphones
print("🔍 Available Microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")

# Select the correct microphone
MIC_INDEX = 0  # Ensure this is correct based on the printed list
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=MIC_INDEX)

recognizer.energy_threshold = 300  # Lowered threshold
recognizer.dynamic_energy_threshold = True



def listen_for_command():
    global exit_flag
    with mic as source:
        print("🎤 Adjusting for background noise... (only once)")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce duration

    while not exit_flag:
        print("🎧 Listening... Say 'what's in front of me?'")
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print("🔊 You said:", command)

                # Allow partial phrase matches
                if "what's in front" in command:
                    print("✅ Detected trigger phrase! Processing command...")
                    process_speech_command()
                else:
                    print(f"❌ Phrase not matched. Heard: {command}")

                time.sleep(1)

            except sr.WaitTimeoutError:
                print("⏳ Timeout: No speech detected.")
            except sr.UnknownValueError:
                print("❌ Didn't catch that. Try again.")
            except sr.RequestError:
                print("❌ Speech Recognition service error.")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")

# Function to process the speech command
def process_speech_command():
    global detected_objects
    if detected_objects:
        # Prioritize the closest object
        prioritized_object = min(
            detected_objects.items(),
            key=lambda x: {"very close": 1, "near": 2, "far": 3}[x[1][1]]
        )

        obj_name, (direction, distance) = prioritized_object
        announcement = f"I see {obj_name} {direction}, {distance}."
        print("📢 Announcing:", announcement)

        if not speaking_queue.full():  # Avoid blocking if the queue is full
            speaking_queue.put(announcement)

# Threaded speech function
def speak(text):
    thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True)
    thread.start()

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

        # Convert to grayscale and enhance contrast, helps with text recognition
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Refinements to OCR performance
        results = reader.readtext(
            thresh_roi,
            detail=0,
            contrast_ths=0.5,  # Lower contrast threshold
            text_threshold=0.4,  # Accept lower confidence text
            link_threshold=0.6   # More flexible word linking
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

# Start voice command listener in a separate thread (prevents blocking)
speech_listener_thread = threading.Thread(target=listen_for_command, daemon=True)
speech_listener_thread.start()

detected_objects = {}  # Stores detected objects

with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    prev_time = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        left_region = w // 3
        right_region = 2 * (w // 3)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        results_objects = model(frame)
        detected_objects.clear()  # Reset detected objects each frame

        text_detected = False
        roi = None

        for r in results_objects:
            # frame = r.plot()
            for box in r.boxes:
                class_id = int(box.cls[0])
                object_name = model.model.names[class_id]

                x_center = box.xywh[0][0]  # x_center is already available
                box_width = box.xywh[0][2]  # Width of bounding box

                # Estimate distance
                if box_width > w * 0.5:
                    distance = "very close"
                elif box_width < w * 0.3:
                    distance = "far"
                else:
                    distance = "near"

                # Determine direction
                if x_center < left_region:
                    direction = "on the left"
                elif x_center > right_region:
                    direction = "on the right"
                else:
                    direction = "in the center"

                detected_objects[object_name] = (direction, distance)

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

                #region slightly in front of the fingertip
                roi_size = 150
                roi_width = 150
                roi_height = 75
                x1, y1 = max(0, x - roi_width // 2), max(0, y - roi_height - 20)
                x2, y2 = min(w, x + roi_width // 2), min(h, y - 20)

                roi = frame[y1:y2, x1:x2]

                # Limit OCR calls to every certain number of frames (12 rn)
                if (roi.size > 0 and not ocr_thread_running and
                    frame_count % ocr_frame_skip == 0 and index_tip.y < index_mcp.y and  
                    middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y and pinky_tip.y > index_mcp.y):

                    ocr_thread_running = True  # Mark OCR as running
                    ocr_queue.put(roi)  # Add ROI to the queue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                # Draw a circle at the index finger tip
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        end_time = time.time()
        fps = 1/(end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Hand Pointing Detection & Text Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break


ocr_queue.put(None)
speaking_queue.put(None)

ocr_processing_thread.join()
speech_thread.join()
speech_listener_thread.join()

del mic
cap.release()
cv2.destroyAllWindows()