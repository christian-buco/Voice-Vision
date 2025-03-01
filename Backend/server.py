import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model (ensure "yolov8n.pt" is in your working directory)
model = YOLO("yolov8n.pt")

def process_frame(frame_b64):
    # Decode the base64 string into bytes and open as an image
    img_data = base64.b64decode(frame_b64)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    
    # Convert PIL image to OpenCV format (BGR)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run YOLO model on the frame
    results = model(img_bgr)
    
    # Annotate the frame with the detections
    for r in results:
        img_bgr = r.plot()
        print(r)

    # Encode the processed image to JPEG and then to base64
    _, buffer = cv2.imencode('.jpg', img_bgr)
    processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return processed_frame_b64

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    frame_b64 = data['image']
    try:
        processed_frame_b64 = process_frame(frame_b64)
        return jsonify({'processedFrame': processed_frame_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 at port 5000
    app.run(host='localhost', port=5000)