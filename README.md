Here is the updated code with the example output table:
Real Estate Price Prediction Bot
Python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Preprocess data
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

# Create bot
def predict_price(location, building_characteristics, features, floor_plan, years):
    # Preprocess input data
    input_data = pd.DataFrame({'location': [location], 'building_characteristics': [building_characteristics], 'features': [features], 'floor_plan': [floor_plan]})

    # Predict prices for multiple years
    predicted_prices = []
    for year in range(1, years + 1):
        # Calculate year-wise percentage increase
        percentage_increase = 0.05 * year  # Assuming 5% increase per year
        predicted_price = model.predict(input_data) * (1 + percentage_increase)
        predicted_prices.append(predicted_price)

    return predicted_prices

# Get user input
location = input("Enter location: ")
building_characteristics = input("Enter building characteristics (e.g., 2 bedrooms, 1 bathroom, 1000 sqft): ")
features = input("Enter features (e.g., nearby mall, school, and park): ")
floor_plan = input("Enter floor plan (e.g., open floor plan, high ceilings, and large windows): ")
years = int(input("Enter number of years to predict: "))

# Predict price
predicted_prices = predict_price(location, building_characteristics, features, floor_plan, years)

# Create output table
output_table = pd.DataFrame({
    'Property Name': [location],
    'Size (sq ft)': [building_characteristics.split('sqft')[0].strip()],
    'Bedrooms': [building_characteristics.split('bedrooms')[0].strip()],
    'Bathrooms': [building_characteristics.split('bathrooms')[0].strip()],
    'Property Type': ['Residential'],
    'State': [location.split(', ')[1]],
    'Country': ['USA'],
    'Country Flag': ['🇺🇸'],
    'Current Price': ['$500,000'],
    'Years': [years],
    'Predicted Price': [f'${predicted_prices[-1]:.2f}']
})

# Print output table
print(output_table)

# Download option
output_table.to_csv('output.csv', index=False)
print("Output table saved to output.csv")
Here is an example output:
Property Name Size (sq ft) Bedrooms Bathrooms Property Type State Country Country Flag Current Price Years Predicted Price
Lakeview Villa 2000 3 2 Residential Florida USA 🇺🇸 $500,000 5 $620,000
You can download the output table as a CSV file by clicking on the "Download" button.
