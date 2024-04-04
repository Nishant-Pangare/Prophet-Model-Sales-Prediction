import pandas as pd
from prophet import Prophet
from flask import Flask, request, jsonify
import calendar

# Initialize Flask app
app = Flask(__name__)

# Load the Prophet model
model = Prophet()

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

    # Fit the model to the data
    model.fit(df)

    # Find maximum date within the data
    max_date = df['ds'].max()

    # Adjust start date for predictions to the next month after the maximum date in the data
    start_date = pd.to_datetime(max_date) + pd.DateOffset(months=1)

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

    # Prepare response
    response = {
        "Predicted Sales For Upcoming 12 months": predicted_sales[['Year', 'Month', 'Predicted Sales']].to_dict('records')
    }
    
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=1000)
