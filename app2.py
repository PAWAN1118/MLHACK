import requests

url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Prepare data
X = df_historical[["day"]]
y = df_historical["cases"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace Linear Regression with SVM (SVR)
model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
predicted_cases = model.predict(next_day)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")
