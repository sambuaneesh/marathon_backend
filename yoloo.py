import torch
from PIL import Image
from torchvision.transforms import functional as F
from flask import Flask, request, jsonify

app = Flask(__name__)


def convert_to_custom_format(predictions):
    custom_predictions = []
    for pred in predictions:
        # Extract information from the prediction tensor
        x, y, width, height = pred[0:4]
        class_probabilities = pred[5:]
        class_confidence, class_index = torch.max(class_probabilities, dim=0)

        # If the confidence score is above a certain threshold, consider it a valid detection
        confidence_threshold = 0.5
        if class_confidence > confidence_threshold:
            # Convert YOLO coordinates to (x_min, y_min, x_max, y_max)
            x_min = (x - width / 2).item()
            y_min = (y - height / 2).item()
            x_max = (x + width / 2).item()
            y_max = (y + height / 2).item()

            # Get class label (replace class_index with actual class labels if available)
            class_label = class_index.item()

            # Create a dictionary for the detected object
            detected_object = {
                "class": class_label,
                "confidence": class_confidence.item(),
                "bounding_box": [x_min, y_min, x_max, y_max],
            }

            # Add the detected object to the list of custom predictions
            custom_predictions.append(detected_object)

    return custom_predictions


# Load your YOLOv8 model here
yolo_model = "stages.pt"


@app.route("/predict-yolov8", methods=["POST"])
def predict_yolov8():
    try:
        # Get the image from the POST request
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        # Process the image using your YOLOv8 model
        input_tensor = F.to_tensor(image).unsqueeze(0).cuda()
        with torch.no_grad():
            predictions = yolo_model(input_tensor)

        # Post-process predictions if necessary (e.g., apply non-maximum suppression)
        # Replace the following line with your actual post-processing code if needed.
        # predictions = perform_nms(predictions)

        # Convert YOLOv8 predictions to the desired format (list of dictionaries)
        yolo_predictions = convert_to_custom_format(predictions)

        # Return the predictions as JSON response
        return jsonify(yolo_predictions)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
