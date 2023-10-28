from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("ResNet50V2_eurosat.h5")

# Define class labels
class_labels = [
    "class1",
    "class2",
    "class3",
    "class4",
    "class5",
    "class6",
    "class7",
    "class8",
    "class9",
    "class10",
]


# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to 0-1
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files["image"]
        # Save the image temporarily
        image_path = "temp.jpg"
        image_file.save(image_path)
        # Preprocess the image for prediction
        processed_image = preprocess_image(image_path)
        # Make prediction
        predictions = model.predict(processed_image)
        # Get the predicted class label
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        response = {"success": True, "predicted_class": predicted_class}
    except Exception as e:
        response = {"success": False, "error": str(e)}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
