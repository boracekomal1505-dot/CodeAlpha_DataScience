import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("car_data.csv")

print("Dataset Preview:")
print(data.head())

# Features
X = data[['Year','Present_Price','Kms_Driven']]

# Target
y = data['Selling_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Prediction example
prediction = model.predict([[2020,10,20000]])

print("Predicted Car Price:", prediction)
