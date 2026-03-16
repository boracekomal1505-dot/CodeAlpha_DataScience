import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("advertising.csv")

print("Dataset Preview:")
print(data.head())
# Features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Prediction example
prediction = model.predict([[200, 40, 60]])

print("Predicted Sales:", prediction)