import pandas as pd
from prophet import Prophet
from flask import Flask, request, jsonify
import calendar
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Prophet model
model = joblib.load('Salon Data Prophet Sales Prediction\Prophet_model.pkl')

# Define endpoint for making predictions
@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    # Load data file from request
    data = request.files['data']
    
    # Check file type
    if data.filename.endswith('.csv'):
        df = pd.read_csv(data)
    elif data.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data)
    else:
        return jsonify({"error": "Unsupported file format. Please provide a CSV or Excel file."})
    
    # Rename columns for Prophet model
    if {"Date", "Total Sales"}.issubset(df.columns):
        df.rename(columns={"Date": "ds", "Total Sales": "y"}, inplace=True)
    elif {"ds", "y"}.issubset(df.columns):
        pass
    else:
        return jsonify({"error": "Required columns 'Date' and 'Total Sales' or 'ds' and 'y' not found."})

    # Make predictions
    predictions = make_predictions(df, model)

    return jsonify(predictions)

def make_predictions(df, model):
    # # Find maximum date within the data
    # max_date = df['ds'].max()
    # Convert the 'Date' column to datetime format
    df['ds'] = pd.to_datetime(df['ds'])

# Find the maximum date value
    max_date = df['ds'].max()

# Print the maximum date value
    #print("Maximum date value in the CSV file:", max_date)

    # Adjust start date for predictions to the next month after the maximum date in the data
    start_date = pd.to_datetime(max_date, dayfirst=True) + pd.DateOffset(months=1)
    #print("Start Date of prediction is: ",start_date)

    # Create future dataframe for the next 12 months from the adjusted start date
    future_dates = pd.DataFrame(pd.date_range(start=start_date, periods=12, freq='M'), columns=['ds'])

    # Make predictions for the next 12 months
    forecast = model.predict(future_dates)

    # Extract predicted sales values and corresponding months and years
    predicted_sales = forecast[['ds', 'yhat']].copy()
    predicted_sales.rename(columns={"yhat": "Predicted Sales"}, inplace=True)
    predicted_sales['Month'] = predicted_sales['ds'].dt.month.map(lambda x: calendar.month_abbr[x])
    predicted_sales['Year'] = predicted_sales['ds'].dt.year

    # Round predicted sales values to 2 decimal places
    predicted_sales['Predicted Sales'] = predicted_sales['Predicted Sales'].round(2)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((df['y'] - forecast['yhat'])**2)).round(2)

    # Prepare response
    response = {
        "Predicted Sales For Next 12 Months": predicted_sales[['Year', 'Month', 'Predicted Sales']].to_dict('records'),
        # "RMSE": rmse
    }
    
    return response

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=1000)
