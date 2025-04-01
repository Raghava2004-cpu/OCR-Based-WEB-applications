import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont  # Import ImageFont for better labels
import io
from ultralytics import YOLO

# --- Load YOLO Model ---
MODEL_PATH = 'model/char.pt'

try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Function for Gradio ---
def predict(image):
    if model is None:
        return "Model is not loaded properly."

    try:
        img = Image.fromarray(image).convert('RGB')  # Convert to PIL Image
        results = model(img)  # Perform inference

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # Load a default font for text

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    label = model.model.names[int(box.cls)]  # Get class label
                    confidence = float(box.conf[0])  # Get confidence score

                    # Draw bounding box and text
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    text = f"{label} ({confidence:.2f})"
                    draw.text((x1, y1 - 10), text, fill="red", font=font)

        return img  # Return the image with drawn boxes

    except Exception as e:
        return f"Error during prediction: {e}"

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Image(label="Image with Predictions"),
    title="YOLO Object Detection",
    description="Upload an image to see object detection predictions using a YOLO model.",
)

iface.launch()
