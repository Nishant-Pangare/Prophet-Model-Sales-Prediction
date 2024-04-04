# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # Read your sales data from Excel
# df = pd.read_excel('Salon Data Prophet Sales Prediction\Salon New Data.xlsx')

# # Check for missing values and handle them appropriately (if needed)
# missing_values = df['Date'].isnull().sum()
# if missing_values > 0:
#   print(f"There are {missing_values} missing values in the 'Date' column.")
#   # Handle missing values (e.g., remove rows, impute using techniques like forward fill)
# else:
#   print("No missing values in the Date Column.")

# # Ensure the 'Date' column is in datetime format
# if not pd.api.types.is_datetime64_dtype(df['Date']):
#   df['Date'] = pd.to_datetime(df['Date'])

# # Rename columns for Prophet model
# df.rename(columns={"Date": "ds", "Total Sales": "y"}, inplace=True)

# # Select relevant columns (optional, include "Prod Number" for product filtering if needed)
# sales_data = df[["ds", "y"]]

# # Define a function to calculate mean squared error
# def custom_scoring(y_true, y_pred):
#   return mean_squared_error(y_true, y_pred)

# # Define the Prophet model (uses default hyperparameters)
# model = Prophet()

# # Fit the model to your sales data
# model.fit(sales_data)

# # Define the number of periods (months) to predict for future months
# periods = 12

# # Define the cutoff date (optional, set it to the last date in your data minus 1 day)
# cutoff_date = sales_data['ds'].max() - pd.DateOffset(days=1)

# # Create the future dataframe
# future_dates = model.make_future_dataframe(periods=periods)

# # Filter the future dataframe to include dates after the cutoff date (if using cutoff)
# if cutoff_date is not None:
#   future_dates = future_dates[future_dates['ds'] > cutoff_date]

# # Make predictions for the upcoming months
# forecast_upcoming = model.predict(future_dates)

# # Rename columns for clarity
# forecast_upcoming.rename(columns={
#   "ds": "Date",
#   "yhat": "Predicted Sales",
#   "yhat_lower": "Lower Sales Bound",
#   "yhat_upper": "Upper Sales Bound"
# }, inplace=True)

# # Plot predictions for the upcoming months timeframe
# plt.figure(figsize=(10, 6))
# plt.plot(forecast_upcoming['Date'], forecast_upcoming['Predicted Sales'], label='Predicted Sales', color='green')
# plt.fill_between(forecast_upcoming['Date'], forecast_upcoming['Lower Sales Bound'], forecast_upcoming['Upper Sales Bound'], color='lightgreen', alpha=0.5)
# plt.title('Predicted Sales for Upcoming Months')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()

# # Calculate Mean Squared Error (MSE)
# # Ensure that the lengths of actual and predicted sales data are the same
# sales_data_subset = sales_data.iloc[:len(forecast_upcoming)]
# mse = mean_squared_error(sales_data_subset['y'], forecast_upcoming['Predicted Sales'])
# print("Mean Squared Error (MSE):", mse)

# print("Shape of actual sales data:", sales_data_subset.shape)
# print("Shape of predicted sales data:", forecast_upcoming.shape)

# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import calendar

# # Read your sales data from Excel
# df = pd.read_excel('Salon Data Prophet Sales Prediction\Salon New Data.xlsx')

# # Check for missing values and handle them appropriately (if needed)
# missing_values = df['Date'].isnull().sum()
# if missing_values > 0:
#     print(f"There are {missing_values} missing values in the 'Date' column.")
#     # Handle missing values (e.g., remove rows, impute using techniques like forward fill)
# else:
#     print("No missing values in the Date Column.")

# # Ensure the 'Date' column is in datetime format
# if not pd.api.types.is_datetime64_dtype(df['Date']):
#     df['Date'] = pd.to_datetime(df['Date'])

# # Rename columns for Prophet model
# df.rename(columns={"Date": "ds", "Total Sales": "y"}, inplace=True)

# # Select relevant columns (optional, include "Prod Number" for product filtering if needed)
# sales_data = df[["ds", "y"]]

# # Define the Prophet model
# model = Prophet()

# # Fit the model to your sales data
# model.fit(sales_data)

# # Define the number of periods (months) to predict for future months
# periods = 12

# # Create the future dataframe to include all months of 2024 and 2025
# future_dates_2024 = pd.DataFrame(pd.date_range(start='2024-01-01', end='2024-12-31', freq='M'), columns=['ds'])
# future_dates_2025 = pd.DataFrame(pd.date_range(start='2025-01-01', end='2025-12-31', freq='M'), columns=['ds'])

# # Make predictions for the entire years of 2024 and 2025
# forecast_2024 = model.predict(future_dates_2024)
# forecast_2025 = model.predict(future_dates_2025)

# # Extract predicted sales values, upcoming month, and year for 2024
# predicted_sales_2024 = forecast_2024[['ds', 'yhat']].copy()
# predicted_sales_2024.rename(columns={"yhat": "Predicted Sales"}, inplace=True)
# predicted_sales_2024['Month'] = predicted_sales_2024['ds'].dt.month.map(lambda x: calendar.month_abbr[x])
# predicted_sales_2024['Year'] = predicted_sales_2024['ds'].dt.year

# # Extract predicted sales values, upcoming month, and year for 2025
# predicted_sales_2025 = forecast_2025[['ds', 'yhat']].copy()
# predicted_sales_2025.rename(columns={"yhat": "Predicted Sales"}, inplace=True)
# predicted_sales_2025['Month'] = predicted_sales_2025['ds'].dt.month.map(lambda x: calendar.month_abbr[x])
# predicted_sales_2025['Year'] = predicted_sales_2025['ds'].dt.year

# # Display the predicted sales values, upcoming month, and year for 2024
# print("Predicted Sales for the Year 2024:")
# print(predicted_sales_2024[['Year', 'Month', 'Predicted Sales']])

# # Display the predicted sales values, upcoming month, and year for 2025
# print("Predicted Sales for the Year 2025:")
# print(predicted_sales_2025[['Year', 'Month', 'Predicted Sales']])

# # Plot predictions for the entire years of 2024 and 2025
# plt.figure(figsize=(10, 6))

# # Plot predicted sales for 2024
# plt.plot(predicted_sales_2024['ds'], predicted_sales_2024['Predicted Sales'], label='Predicted Sales 2024', color='green')

# # Plot predicted sales for 2025
# plt.plot(predicted_sales_2025['ds'], predicted_sales_2025['Predicted Sales'], label='Predicted Sales 2025', color='blue')

# plt.title('Predicted Sales for the Years 2024 and 2025')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()

# # Calculate Mean Squared Error (MSE) for both 2024 and 2025
# mse_2024 = mean_squared_error(sales_data['y'][:len(forecast_2024)], forecast_2024['yhat'])
# mse_2025 = mean_squared_error(sales_data['y'][:len(forecast_2025)], forecast_2025['yhat'])

# print("Mean Squared Error (MSE) for 2024:", mse_2024)
# print("Mean Squared Error (MSE) for 2025:", mse_2025)









