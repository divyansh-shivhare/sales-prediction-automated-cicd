import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("data/add.csv")
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=55)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model trained successfully for first time checking")
print(f"R2 Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
