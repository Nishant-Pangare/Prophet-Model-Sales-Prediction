import pandas as pd

# Load the data
df = pd.read_csv(r"C:\Users\HP\Desktop\Salon Data New.csv")  # Replace 'your_data.xlsx' with the path to your Excel file

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with missing or invalid dates
df.dropna(subset=['Date'], inplace=True)

# Sort the DataFrame by Date column in ascending order
df.sort_values(by='Date', inplace=True)

# Rename columns to match Prophet's requirements
# Rename columns to match Prophet's requirements
df.rename(columns={'Date': 'ds', 'Total Sales': 'y'}, inplace=True)

# Export the preprocessed DataFrame to a CSV file
df.to_csv('Salon Preprocessed Data.csv', index=False)
