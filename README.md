# CAR PRICE PREDICTION SYSTEM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data_records = {
    'brand': ['Toyota', 'Honda', 'BMW', 'Toyota', 'Honda', 'BMW', 'Ford', 'Tesla', 'Hyundai', 'Toyota', 'Honda', 'BMW'],
    'model_name': ['Corolla', 'Civic', 'X5', 'Camry', 'Accord', '3 Series', 'Focus', 'Model 3', 'Elantra', 'RAV4', 'Pilot', 'X3'],
    'manufacture_year': [2015, 2016, 2017, 2015, 2016, 2018, 2017, 2020, 2018, 2019, 2017, 2019],
    'distance_covered': [50000, 40000, 30000, 60000, 50000, 20000, 55000, 15000, 40000, 25000, 45000, 35000],
    'fuel_category': ['Petrol', 'Diesel', 'Diesel', 'Petrol', 'Petrol', 'Diesel', 'Petrol', 'Electric', 'Petrol', 'Petrol', 'Diesel', 'Diesel'],
    'car_type': ['Sedan', 'Sedan', 'SUV', 'Sedan', 'Sedan', 'SUV', 'Hatchback', 'Sedan', 'Sedan', 'SUV', 'SUV', 'SUV'],
    'owners_count': [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1],
    'service_history': [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'car_value': [15000, 16000, 30000, 14000, 18000, 35000, 12000, 40000, 17000, 25000, 20000, 32000],
}

dataframe = pd.DataFrame(data_records)

processed_data = pd.get_dummies(
    dataframe,
    columns=['brand', 'model_name', 'fuel_category', 'car_type'],
    drop_first=True
)

features = processed_data.drop('car_value', axis=1)
target = processed_data['car_value']

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

price_predictor = LinearRegression()
price_predictor.fit(features_train, target_train)

predicted_values = price_predictor.predict(features_test)
error = mean_absolute_error(target_test, predicted_values)
print(f"Mean Absolute Error: ${error:.2f}")

new_vehicle = {
    'manufacture_year': 2021,
    'distance_covered': 10000,
    'owners_count': 1,
    'service_history': 1,
    'brand_BMW': 0, 'brand_Ford': 0, 'brand_Honda': 0, 'brand_Hyundai': 0, 'brand_Tesla': 1, 'brand_Toyota': 0,
    'model_name_3 Series': 0, 'model_name_Accord': 0, 'model_name_Camry': 0, 'model_name_Civic': 0, 'model_name_Corolla': 0,
    'model_name_Elantra': 0, 'model_name_Focus': 0, 'model_name_Model 3': 1, 'model_name_RAV4': 0, 'model_name_X3': 0, 'model_name_X5': 0,
    'fuel_category_Diesel': 0, 'fuel_category_Electric': 1, 'fuel_category_Petrol': 0,
    'car_type_Hatchback': 0, 'car_type_Sedan': 1, 'car_type_SUV': 0,
}

new_vehicle_data = pd.DataFrame([new_vehicle])

missing_columns = set(features.columns) - set(new_vehicle_data.columns)
for column in missing_columns:
    new_vehicle_data[column] = 0
new_vehicle_data = new_vehicle_data[features.columns]

estimated_price = price_predictor.predict(new_vehicle_data)[0]
print(f"Estimated Price for the new vehicle: ${estimated_price:.2f}")


