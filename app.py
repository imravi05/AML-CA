# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
from prophet import Prophet # Ensure Prophet is imported for joblib to load it correctly

app = Flask(__name__)

# Define the path to the trained model
MODEL_PATH = 'prophet_rainfall_model.joblib'

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training script was run.")
    model = None # Set model to None to handle errors gracefully
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """
    Renders the home page with the input form for rainfall prediction.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Expects a 'year' in the form data, makes a prediction, and returns the result.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Get the year from the form input
        year_str = request.form['year']
        prediction_year = int(year_str)

        # Create a DataFrame for the future prediction date required by Prophet
        # Prophet expects a DataFrame with a 'ds' column (datetime)
        future_df = pd.DataFrame({'ds': [pd.to_datetime(f'{prediction_year}-01-01')]})

        # Make prediction
        forecast = model.predict(future_df)

        # Extract the predicted rainfall value (yhat)
        predicted_rainfall = forecast['yhat'].iloc[0]

        # Format the output
        result = {
            'year': prediction_year,
            'predicted_rainfall': round(predicted_rainfall, 2) # Round to 2 decimal places
        }
        return render_template('index.html', prediction=result)

    except ValueError:
        return render_template('index.html', error="Invalid year. Please enter a valid number.", prediction=None)
    except KeyError:
        return render_template('index.html', error="Missing 'year' input. Please provide a year.", prediction=None)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction: {str(e)}", prediction=None)

if __name__ == '__main__':
    # Ensure the model is trained and saved before running the Flask app.
    # In a real deployment, this would be handled by a build process.
    # For this demonstration, we assume the model training script has been run.
    app.run(debug=True) # debug=True is good for development, disable in production
