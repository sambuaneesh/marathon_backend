from flask import Flask, request, jsonify
from roboflow import Roboflow
from flask_cors import CORS
import argparse
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Initialize the Roboflow client for paddy type classification
rf_type = Roboflow(api_key="kiU6DLRhU8ITt8utfzMi")
project_type = rf_type.workspace().project("rice-classification-4dzut")
model_type = project_type.version(2).model

# Initialize the Roboflow client for paddy stage prediction
rf_stage = Roboflow(api_key="kiU6DLRhU8ITt8utfzMi")
project_stage = rf_stage.workspace().project("ricestage")
model_stage = project_stage.version(1).model


# Define the "/paddy-type" endpoint for classification prediction
@app.route("/paddy-type", methods=["POST"])
def predict_paddy_type():
    try:
        # Get the image from the POST request
        image_file = request.files["image"]

        # Save the uploaded image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file.save(temp_file.name)

        # Perform classification prediction using Roboflow model
        response = model_type.predict(temp_file.name)
        print(response.json())
        # Close and remove the temporary file
        temp_file.close()
        os.unlink(temp_file.name)

        # Extract class predictions and find the class with the highest confidence
        predictions_list = response.json().get("predictions", [])
        if predictions_list:
            first_prediction = predictions_list[0]
            predictions = first_prediction.get("predictions", {})
            if predictions:
                max_confidence_class = max(
                    predictions, key=lambda k: predictions[k]["confidence"]
                )
                return max_confidence_class

        # Handle the case where no predictions are returned
        return "No predictions available."

    except Exception as e:
        return (
            str(e),
            500,
        )  # Return the error message and 500 status code for internal server error


# Define the "/paddy-stage" endpoint for stage prediction
@app.route("/paddy-stage", methods=["POST"])
def predict_paddy_stage():
    try:
        # Get the image from the POST request
        image_file = request.files["image"]

        # Save the uploaded image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file.save(temp_file.name)

        # Perform stage prediction using Roboflow model
        response = model_stage.predict(temp_file.name)

        # Close and remove the temporary file
        temp_file.close()
        os.unlink(temp_file.name)

        # Return the stage prediction result as JSON response
        # return jsonify(response.json())
        resp = str(response.json())
        result = int(resp[resp.index("top") + 7 : resp.index("top") + 8])
        if result == 0:
            return "Seedling Stage (1/5)"
        elif result == 1:
            return "Tillering Stage (2/5)"
        elif result == 2:
            return "Heading/Flowering Stage (3/5)"
        elif result == 3:
            return "Milky Stage (4/5)"
        elif result == 4:
            return "Ripening Stage (5/5)"
        else:
            return "Heading/Flowering Stage (3/5)"

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # Parse command line arguments to get custom port number
    parser = argparse.ArgumentParser(description="Custom Flask App")
    parser.add_argument(
        "--port", type=int, default=5003, help="Port number for the Flask app"
    )
    args = parser.parse_args()

    # Run the Flask app with the custom port
    app.run(port=args.port, debug=True)
