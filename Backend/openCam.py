import cv2
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO
import time
import threading
import queue

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Queue for speech announcements
speaking_queue = queue.Queue()

# Function to announce detected objects
def speak_text():
    while True:
        text = speaking_queue.get()
        if text is None:
            continue  # Skip processing if None is retrieved
        print(f"🔊 Speaking: {text}")  # Debugging print
        engine.say(text)
        engine.runAndWait()


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

recognizer.energy_threshold = 100  # Lowered threshold
recognizer.dynamic_energy_threshold = True

# Auto-detect camera
cap = None
for i in range(10):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        print(f"📷 Camera found at index {i}")
        cap = temp_cap
        break
    temp_cap.release()

if cap is None or not cap.isOpened():
    print("❌ Error: Could not open the camera.")
    exit()

# Function to continuously listen for speech commands (in a separate thread)
def listen_for_command():
    while True:
        with mic as source:
            print("🎤 Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased to 2 sec

            print("🎧 Listening... Say 'what's in front of me?'")
            try:
                audio = recognizer.listen(source, timeout=10)  # Increased timeout
                command = recognizer.recognize_google(audio).lower()
                print("🔊 You said:", command)

                if "what's in front of me" in command:
                    process_speech_command()

                # Add a small delay to prevent rapid looping in case of continuous speech
                time.sleep(1)

            except sr.WaitTimeoutError:
                print("⏳ Timeout: No speech detected.")
            except sr.UnknownValueError:
                print("❌ Didn't catch that. Try again.")
            except sr.RequestError:
                print("❌ Speech Recognition service error.")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")  # Ensures the loop never exits

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

# Start voice command listener in a separate thread (prevents blocking)
speech_listener_thread = threading.Thread(target=listen_for_command, daemon=True)
speech_listener_thread.start()

detected_objects = {}  # Stores detected objects

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera feed stopped! Exiting program...")
        break

    height, width, _ = frame.shape
    left_region = width // 3
    right_region = 2 * (width // 3)

    results = model(frame)
    detected_objects.clear()  # Reset detected objects each frame

    for r in results:
        frame = r.plot()
        for box in r.boxes:
            class_id = int(box.cls[0])
            object_name = model.model.names[class_id]

            x_center = box.xywh[0][0]  # x_center is already available
            box_width = box.xywh[0][2]  # Width of bounding box

            # Estimate distance
            if box_width > width * 0.5:
                distance = "very close"
            elif box_width < width * 0.3:
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

    # print("👀 Detected objects:", detected_objects)
    cv2.imshow("AI Glasses Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting program via user input (q)")
        break

# Keep threads running
speech_thread.join()
speech_listener_thread.join()

print("❌ Exiting: Cleaning up camera and windows.")
cap.release()
cv2.destroyAllWindows()
