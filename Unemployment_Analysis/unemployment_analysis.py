import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Unemployment.csv")

# Show first rows
print(data.head())

# Plot unemployment rate
data["Estimated Unemployment Rate (%)"].plot()

plt.title("Unemployment Rate Trend")
plt.xlabel("Index")
plt.ylabel("Unemployment Rate")
plt.show()