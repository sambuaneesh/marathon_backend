from flask import Flask, request, jsonify
from roboflow import Roboflow
import argparse
import os
import tempfile

app = Flask(__name__)

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

        # Close and remove the temporary file
        temp_file.close()
        os.unlink(temp_file.name)

        # Return the classification prediction result as JSON response
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)})


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
        return jsonify(response.json())

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
