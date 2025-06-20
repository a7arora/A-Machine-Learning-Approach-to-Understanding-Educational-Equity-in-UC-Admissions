import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt


df = pd.read_csv("EducationalEquity.csv")
df = df.drop(columns=["Unnamed: 1"])


X_raw = df[['SBAC Percentile']].values  # 2D array
y_raw = df['UCI Admission'].values.reshape(-1, 1)  # 2D for scaler


X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.3, random_state=42
)


poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly_raw = poly.fit_transform(X_train_raw)
X_test_poly_raw = poly.transform(X_test_raw)


scaler_X = StandardScaler()
scaler_y = StandardScaler()


X_train = scaler_X.fit_transform(X_train_poly_raw)
X_test = scaler_X.transform(X_test_poly_raw)


y_train = scaler_y.fit_transform(y_train_raw)
y_test = scaler_y.transform(y_test_raw)


lr = LinearRegression()
lr.fit(X_train, y_train.ravel())


y_pred_norm = lr.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test).ravel()


rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print(f"Polynomial Regression (Degree 3) RMSE: {rmse:.3f}")
print(f"Polynomial Regression (Degree 3) R² Score: {r2:.3f}")
print(f"Model Coefficients: {lr.coef_}")
print(f"Model Intercept (normalized space): {lr.intercept_:.4f}")

results_df = pd.DataFrame({
    "SBAC Percentile": X_test_raw.ravel(),
    "Predicted UCI Admission Rate (%)": y_pred,
    "Actual UCI Admission Rate (%)": y_test_orig
})

# Print table
print("\n--- Prediction Results ---")
print(results_df.round(2).to_string(index=False))
results_df.to_csv("uci_admission_predictions.csv", index=False)

# Plot actual vs predicted (original scale)
plt.figure(figsize=(8,5))
plt.scatter(X_test_raw, y_test_orig, label="Actual", alpha=0.7)
plt.scatter(X_test_raw, y_pred, label="Predicted", alpha=0.7)
plt.xlabel("SBAC Percentile")
plt.ylabel("UCI Admission Rate")
plt.title("Polynomial Regression (Degree 3) Predictions")
plt.legend()
plt.grid(True)
plt.show()

sbac_percentiles = X_test_raw.ravel()

