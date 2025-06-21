import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle


app = Flask(__name__)


MODEL_DIR = './' 
@app.route("/", methods=["GET"]) 
def index():
    """Renders the initial HTML form page."""
    return render_template("htmml.html")

@app.route("/predict", methods=["POST"]) # Separate route for prediction API
def predict():
    """Receives form data, loads the selected model, makes a prediction, and returns it as JSON."""
    try:
        # Get data from the form
        data = request.form.to_dict()

        # Get the selected model filename
        selected_model_filename = data.get('model_name', 'house_price_model.pkl') # Default model

        # Basic security check for model path
        model_path = os.path.join(MODEL_DIR, selected_model_filename)
        if not os.path.exists(model_path) or not os.path.isfile(model_path) or not model_path.endswith('.pkl'):
            # Return an error if the file doesn't exist or isn't a .pkl
            return jsonify({'error': 'Invalid or unlocatable model selected.'}), 400

        # Load the selected model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            # Catch specific errors during model unpickling
            return jsonify({'error': f'Failed to load model {selected_model_filename}: {str(e)}'}), 500


        bedrooms = float(data.get("bedrooms", 0))
        bathrooms = float(data.get("bathrooms", 0.0))
        sqft_living = float(data.get("sqft_living", 0.0))
        sqft_lot = float(data.get("sqft_lot", 0.0))
        floors = float(data.get("floors", 0.0))
        waterfront = int(data.get("waterfront", 0))
        view = int(data.get("view", 0))
        condition = int(data.get("condition", 0))
        sqft_above = float(data.get("sqft_above", 0.0))
        sqft_basement = float(data.get("sqft_basement", 0.0))
        yr_renovated = int(data.get("yr_renovated", 0))
        statezip = int(data.get("statezip", 0))
        yr_renovated_YN = int(data.get("yr_renovated_YN", 0))
        basement_YN = int(data.get("basement_YN", 0))
        building_age = int(data.get("building_age", 0))

            # Order must match model training
        features = np.array([[
            bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                waterfront, view, condition, sqft_above, sqft_basement,
                yr_renovated, statezip, yr_renovated_YN, basement_YN,
                building_age
            ]])

        prediction = round(model.predict(features)[0], 2)

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        # Log the error for debugging purposes
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        # Return a JSON error message
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
