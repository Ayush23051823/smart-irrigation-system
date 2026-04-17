import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# 1. LOAD + PREPROCESS
# -------------------------------
df = pd.read_csv("irrigation_prediction.csv")
df = df.dropna()

print("Original Columns:\n", df.columns)

# Convert categorical to numeric
df = pd.get_dummies(df)

print("\nColumns After Encoding:\n", df.columns)

# -------------------------------
# SELECT TARGET COLUMN SAFELY
# -------------------------------
target_cols = [col for col in df.columns if "Irrigation_Need" in col]

if len(target_cols) == 0:
    raise ValueError("❌ No Irrigation_Need column found after encoding")

target_col = target_cols[0]  # usually Irrigation_Need_Yes

print("\nTarget Column Used:", target_col)

# -------------------------------
# IMPORTANT: SAME FEATURES AS UI
# -------------------------------
selected_features = ["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Humidity"]

# Check if all features exist
for col in selected_features:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

X = df[selected_features]
y = df[target_col]

# -------------------------------
# 2. CLASSIFICATION MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nClassification Accuracy:", acc)

# -------------------------------
# 3. REGRESSION MODEL (WATER)
# -------------------------------
df["Water_Amount"] = (1 - df["Soil_Moisture"]) * 100

X_reg = df[selected_features]
y_reg = df["Water_Amount"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xr_train, yr_train)

yr_pred = reg.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
print("Water Prediction MAE:", mae)

# -------------------------------
# 4. FEATURE IMPORTANCE
# -------------------------------
importances = clf.feature_importances_

plt.figure()
plt.barh(selected_features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("graph_feature_importance.png")
plt.show()

# -------------------------------
# 5. CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("graph_confusion_matrix.png")
plt.show()

# -------------------------------
# 6. KALMAN FILTER
# -------------------------------
class KalmanFilter:
    def __init__(self):
        self.x = 0.5
        self.P = 1
        self.Q = 0.01
        self.R = 0.05

    def predict(self):
        self.P += self.Q

    def update(self, z):
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

kf = KalmanFilter()

# -------------------------------
# 7. TIME SERIES SIMULATION
# -------------------------------
moisture_list = []
water_list = []
decision_list = []

for i in range(20):
    sample = X.iloc[i]

    pred = clf.predict([sample])[0]
    water = reg.predict([sample])[0]

    sensor = sample["Soil_Moisture"]

    kf.predict()
    updated = kf.update(sensor)

    moisture_list.append(updated)
    water_list.append(water)
    decision_list.append(pred)

# -------------------------------
# 8. RESEARCH GRAPHS
# -------------------------------
t = np.arange(len(moisture_list))

plt.figure()
plt.plot(t, moisture_list, marker='o')
plt.title("Soil Moisture vs Time")
plt.xlabel("Time Step")
plt.ylabel("Moisture")
plt.grid()
plt.savefig("graph_moisture.png")
plt.show()

plt.figure()
plt.plot(t, water_list, marker='s')
plt.title("Water Recommendation vs Time")
plt.xlabel("Time Step")
plt.ylabel("Water (mm)")
plt.grid()
plt.savefig("graph_water.png")
plt.show()

plt.figure()
plt.step(t, decision_list)
plt.title("Irrigation Decision (0 = No, 1 = Yes)")
plt.xlabel("Time Step")
plt.ylabel("Decision")
plt.grid()
plt.savefig("graph_decision.png")
plt.show()

plt.figure()
plt.plot(t, moisture_list, label="Moisture")
plt.plot(t, water_list, label="Water")
plt.legend()
plt.title("Moisture vs Water")
plt.grid()
plt.savefig("graph_combined.png")
plt.show()

# -------------------------------
# 9. SAVE MODELS
# -------------------------------
pickle.dump(clf, open("model.pkl", "wb"))
pickle.dump(reg, open("water_model.pkl", "wb"))

print("\n✅ Models saved successfully!")