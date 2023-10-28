from server import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = "best_model.h5"  # Change this to the path of your .h5 model file
model = load_model(model_path)

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
        img = Image.open(image_file).convert("RGB")
        # Resize the image to (480, 480)
        img = img.resize((480, 480))
        # Convert the image to numpy array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the pixel values to be between 0 and 1

        # Make predictions
        with tf.device("/cpu:0"):  # Ensure the model runs on CPU
            predictions = model.predict(x)

        # Get the predicted label
        predicted_index = np.argmax(predictions[0])
        predicted_label = id2label[str(predicted_index)]

        print("Prediction successful.")
        return jsonify({"success": True, "predicted_label": predicted_label})
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000))
