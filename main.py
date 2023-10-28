from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import argparse
import os

app = Flask(__name__)

# Load the pre-trained model and move it to the CPU
model = torch.load("PaddyDoctorResnet.pt", map_location=torch.device("cpu"))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Define the mapping from prediction indices to class labels
class_labels = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]
id2label = {str(i): label for i, label in enumerate(class_labels)}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files["image"]
        # Open the image using Pillow (PIL)
        image = Image.open(image_file).convert("RGB")
        # Apply the transformation
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor  # Move data to CPU (no .to('cpu') needed)
        # Make predictions
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
        # Get the predicted label
        predicted_label = id2label[str(pred.item())]
        print("Prediction successful.")
        return jsonify({"success": True, "predicted_label": predicted_label})
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port number for the Flask application")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
