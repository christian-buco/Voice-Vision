import cv2
import easyocr
import pyttsx3
import threading
import time
import numpy as np

# Initialize EasyOCR and pyttsx
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()

# threaded speak function
def speak(text):
    thread = threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()))
    thread.start()

cap = cv2.VideoCapture(0)  

# Track last spoken text
last_spoken_text = ""  

# Cooldown for repeating the same text. 
last_spoken_time = 0    
cooldown_seconds = 5    # 5 Second cooldown 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Instead of finger, detect text within this box
    roi_size = 200
    x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
    x2, y2 = (w + roi_size) // 2, (h + roi_size) // 2
    roi = frame[y1:y2, x1:x2]

    if roi.size > 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
        results = reader.readtext(gray_roi, detail=0)

        # If text is detected
        if results:  
            detected_text = " ".join(results).strip()
            cv2.putText(frame, f"Text: {detected_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            current_time = time.time()

            # Only speak if it's a new text OR if enough time has passed
            if detected_text and (detected_text != last_spoken_text or current_time - last_spoken_time > cooldown_seconds):
                speak(detected_text)
                # Update for cooldown
                last_spoken_text = detected_text  
                last_spoken_time = current_time   

    # To see the detection area, draw rectangle. 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Text Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
