import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("EducationalEquity.csv")


df = df.drop(columns=["Unnamed: 1"])


X = df[['UC/CSU eligible', 'AP Participation']]
y = df['SBAC Percentile']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


results = pd.DataFrame({
    "School Name": df.loc[X_test.index, "School Name"],
    "Actual SBAC": y_test.values,
    "Predicted SBAC": y_pred
})


results["Actual SBAC"] = results["Actual SBAC"].round(1)
results["Predicted SBAC"] = results["Predicted SBAC"].round(1)

print("\nPredictions on Test Set:")
print(results.sort_values("School Name").to_string(index=False))
results.to_csv("rf_sbac_predictions_2.csv", index=False)
print("\nPredictions exported to 'rf_sbac_predictions_2.csv'")

print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R² Score: {r2:.2f}")


feature_importances = rf.feature_importances_
for name, importance in zip(X.columns, feature_importances):
    print(f"{name} importance: {importance:.3f}")
