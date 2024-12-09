import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create a dummy dataset for testing
X_dummy = np.array([
    [1500, 100, 2000, 100, 6, 0],
    [2000, 150, 2500, 150, 8, 50],
    [2500, 200, 3000, 200, 10, 100]
])
y_dummy = np.array([70, 60, 50])  # Dummy sustainability scores

# Fit a StandardScaler
scaler = StandardScaler()
X_dummy_scaled = scaler.fit_transform(X_dummy)

# Train a dummy Linear Regression model
model = LinearRegression()
model.fit(X_dummy_scaled, y_dummy)

# Save the scaler to a pickle file
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the model to a pickle file
with open("sustainability_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Dummy model and scaler saved successfully.")
