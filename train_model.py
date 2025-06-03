import pandas as pd
from prophet import Prophet
import joblib
import os

# Define the path for the uploaded file
file_path = 'rainfall_area.csv'

# --- 1. Data Loading and Preparation ---
print("Loading data...")
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print("Original DataFrame head:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    exit()

# Select relevant columns: 'YEAR' as the time series index and 'ANNUAL' as the target variable.
# Prophet requires column names 'ds' for datetime and 'y' for the target variable.
df_prophet = df[['YEAR', 'ANNUAL']].copy()
df_prophet.rename(columns={'YEAR': 'ds', 'ANNUAL': 'y'}, inplace=True)

# Convert 'ds' (YEAR) to datetime objects. Prophet requires this format.
# We'll set it to the start of the year for consistency.
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

print("\nPrepared DataFrame for Prophet (head):")
print(df_prophet.head())
print(f"Data types after preparation:\n{df_prophet.dtypes}")

# --- 2. Model Training ---
print("\nTraining Prophet model...")
# Initialize Prophet model.
# Since we are predicting annual rainfall, there's no strong yearly seasonality to add.
# However, Prophet can still model trend.
model = Prophet(
    yearly_seasonality=False, # No yearly seasonality for annual data
    weekly_seasonality=False, # Not applicable for yearly data
    daily_seasonality=False   # Not applicable for yearly data
)

# Fit the model to the historical data
model.fit(df_prophet)
print("Prophet model trained successfully.")

# --- 3. Model Saving ---
model_filename = 'prophet_rainfall_model.joblib'
joblib.dump(model, model_filename)
print(f"\nTrained model saved as '{model_filename}'")

# Optional: Verify model loading (for development/testing)
# loaded_model = joblib.load(model_filename)
# print(f"Model loaded successfully for verification: {loaded_model}")

print("\nModel training script finished.")
