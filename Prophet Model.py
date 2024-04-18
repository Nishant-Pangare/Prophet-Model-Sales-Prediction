import pandas as pd
from prophet import Prophet
import numpy as np
import pickle

# Load the data
df = pd.read_csv('Salon Data Prophet Sales Prediction\Salon Preprocessed Data.csv')

# Rename columns for Prophet model
if {"Date", "Total Sales"}.issubset(df.columns):
    df.rename(columns={"Date": "ds", "Total Sales": "y"}, inplace=True)
elif {"ds", "y"}.issubset(df.columns):
    pass
else:
    raise ValueError("Required columns 'Date' and 'Total Sales' or 'ds' and 'y' not found.")

# Convert 'ds' column to datetime format with dayfirst=True to avoid warning
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)

# Initialize and fit the Prophet model
model = Prophet()

# Fit the model
model.fit(df)

# Make predictions for the next 3 months
future_dates = model.make_future_dataframe(periods=36, freq='M') 
forecast = model.predict(future_dates)

# Save the model to a pickle file
with open('Prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Calculate RMSE
rmse = np.sqrt(np.mean((df['y'] - forecast['yhat'])**2)).round(2)
print("RMSE:", rmse)
