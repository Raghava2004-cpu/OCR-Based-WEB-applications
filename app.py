from flask import Flask, render_template, request, redirect
import torch
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import base64
from ultralytics import YOLO  # Ensure this is installed

from PIL import ImageFont

# Load the default font directly
font = ImageFont.load_default()

app = Flask(__name__)

# --- Load YOLO Model ---
MODEL_PATH = 'model/char.pt'  # Corrected relative path
try:
    model = YOLO(MODEL_PATH)  # Updated YOLO model loading
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Function ---
def predict(image_bytes):
    if model is None:
        return None, "Error: Model not loaded."

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        results = model(img)  # Perform inference

        predictions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.model.names[int(box.cls)]
                confidence = float(box.conf[0])
                predictions.append({'label': label, 'confidence': confidence, 'bbox': (x1, y1, x2, y2)})

        # Draw bounding boxes
        draw = ImageDraw.Draw(img)
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            label = f"{pred['label']} ({pred['confidence']:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
            draw.text((x1, y1 - 20), label, fill="red", font=font)

        # Convert to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return predictions, img_base64

    except Exception as e:
        return None, f"Error during prediction: {e}"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(request.url)

    if image_file:
        image_bytes = image_file.read()
        predictions, image_base64 = predict(image_bytes)
        return render_template('result.html', predictions=predictions, image_base64=f"data:image/jpeg;base64,{image_base64}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')