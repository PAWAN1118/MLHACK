import requests

url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated COVID-19 Data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Prepare Data for SVM
X = df_historical[["day"]]
y = df_historical["cases"]

# Feature Scaling (SVM performs better with scaling)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train SVM Model
model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
next_day_scaled = scaler_X.transform(next_day)
predicted_cases_scaled = model.predict(next_day_scaled)

# Convert prediction back to original scale
predicted_cases = scaler_y.inverse_transform(predicted_cases_scaled.reshape(-1, 1))[0][0]

print(f"Predicted cases for Day 31: {int(predicted_cases)}")

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(df_historical["day"], df_historical["cases"], color="blue", label="Actual Cases")
plt.plot(df_historical["day"], scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1)), color="red", label="SVM Prediction")
plt.xlabel("Day")
plt.ylabel("Cases")
plt.title("COVID-19 Case Prediction using SVM")
plt.legend()
plt.show()
