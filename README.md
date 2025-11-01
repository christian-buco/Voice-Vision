# VoiceVision 👁️🎙️  
**An AI-powered assistive system built for Raspberry Pi and desktop environments**  
*Combines computer vision, OCR, and voice feedback to help users detect and understand their surroundings.*

---

## 🧠 Overview
VoiceVision is a collection of Python scripts that bring sight and sound together:
- **Object Detection** – Identify nearby objects and hear their names aloud.  
- **Text Reading** – Point at printed text and have it read back to you in real time.  
- **Voice Feedback** – Hands-free operation through text-to-speech responses.

Each script focuses on a specific capability — from YOLOv8 object detection to fingertip-guided OCR.

---

## 📂 Repository Structure

Backend/ (Fiels to run are in backend, other files such my-app are files not in use)
- openCam.py # Object detection + speech (for laptop webcams)
- testCamCooldown.py # Object detection + speech (for Raspberry Pi, threaded + cooldown)
- testReadTextHand.py # Pointing-based OCR + speech (Pi camera + MediaPipe Hands)
- test.py # OCR + hand tracking test (no speech)
- server.py # Optional Flask API for remote frame processing
- yolov8n.pt # YOLOv8 model weights
- requirements.txt # Core dependencies
- .. README.md # (this file)

---

## 🧩 Features

### 🟢 Object Detection (YOLOv8)
- Real-time detection using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Announces newly detected objects aloud  
- Threaded speech engine to prevent frame drops  
- Distance and direction estimation (left / center / right)

### 🟣 Text Recognition (EasyOCR)
- Detects and reads printed text using [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Targets the area just above your **index finger**
- Adjustable OCR frequency and thresholds for speed/accuracy balance
- Threaded speech with cooldown to avoid repeating the same text

### 🔵 Raspberry Pi Support
- Compatible with **Picamera2**, but with low fps
- Audio output handled through `pyttsx3` and `espeak`

---

## ⚙️ Installation

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### or manually
```bash
pip install opencv-python ultralytics easyocr mediapipe pyttsx3 flask picamera2
sudo apt-get install -y espeak
```

### 2️⃣ Ensure model weights are present
yolov8n.pt should be in the same directory as the scripts.
(If missing, Ultralytics will download it automatically on first run.)


---

## 🚀 Usage

| Purpose                                 | File                  | Command                      |
| --------------------------------------- | --------------------- | ---------------------------- |
| 🧠 Object detection (laptop webcam)     | `openCam.py`          | `python openCam.py`          |
| 🧠 Object detection (Raspberry Pi)      | `testCamCooldown.py`  | `python testCamCooldown.py`  |
| 🖐️ Point to read text (Raspberry Pi)   | `testReadTextHand.py` | `python testReadTextHand.py` |
| 🧪 OCR + hand tracking test (no speech) | `test.py`             | `python test.py`             |
| 🌐 Run YOLO server (optional)           | `server.py`           | `python server.py`           |


🗣️ Key Customization Points

| Variable               | Description                           | Default |
| ---------------------- | ------------------------------------- | ------- |
| `FORGET_TIME`          | Time (s) before re-announcing objects | 3       |
| `CONFIDENCE_THRESHOLD` | Minimum YOLO confidence for speech    | 0.5     |
| `ocr_frame_skip`       | Frame interval for OCR processing     | 12      |
| `cooldown_seconds`     | Time (s) before re-reading same text  | 5       |

---

## 🧰 Requirements
- Python 3.9+
- Raspberry Pi 4 (or newer) or a desktop/laptop with webcam
- Speaker or headphone output
- Internet only required for initial YOLO model download


## 🧾 License
MIT License © Nicholas Lewis
Feel free to modify, expand, or build upon this project for educational and assistive-tech purposes.


## 💡 Future Ideas
Merge pointing + object detection into one unified “I’m pointing at a chair” mode
Add voice command support (“describe scene”)
Integrate small OLED or AR display for hybrid visual feedback


## Members
- Bryce Chao
- Christian Buco
- Austin Mann
- Nicholas Lewis



