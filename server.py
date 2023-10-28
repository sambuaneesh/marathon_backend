from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms
from keras.models import load_model
from keras.preprocessing import image as keras_image
import numpy as np
import argparse

app = Flask(__name__)
CORS(app)

# Load PyTorch model and move it to the CPU (Disease Classification)
disease_model = torch.load(
    "PaddyDoctorResnet.pt", map_location=torch.device("cpu")
)  # torch_model
disease_model.eval()

# Load Keras model (Land Classification)
land_model = load_model("ResNet50V2_eurosat.h5")  # keras_model

# Define transformations for the PyTorch model
torch_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Define class labels for the Keras model
land_class_labels = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Define class labels for the PyTorch model
disease_class_labels = [
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


# Routes for PyTorch model
@app.route("/predict-disease", methods=["POST"])
def predict_torch():
    try:
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        input_tensor = torch_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = disease_model(input_tensor)
            _, pred = torch.max(output, 1)
        predicted_label = disease_class_labels[pred.item()]
        print("PyTorch Prediction successful.")
        return predicted_label
    except Exception as e:
        print(f"PyTorch Prediction failed: {str(e)}")
        return str(e)


# Routes for Keras model
@app.route("/predict-land", methods=["POST"])
def predict_keras():
    try:
        image_file = request.files["image"]
        img = Image.open(image_file).convert("RGB")
        img = img.resize((64, 64))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        predictions = land_model.predict(x)
        predicted_class_index = np.argmax(predictions)
        predicted_class = land_class_labels[predicted_class_index]
        print("Keras Prediction successful.")
        return predicted_class
    except Exception as e:
        print(f"Keras Prediction failed: {str(e)}")
        return str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port number for the Flask application")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)
