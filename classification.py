import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Convert target labels
# M = Malignant -> 1
# B = Benign -> 0

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

print("Dataset Shape:", df.shape)

# ---------------------------------------------
# Train-Test Split
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------
# Feature Scaling
# ---------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------
# Train Logistic Regression Model
# ---------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ---------------------------------------------
# Predictions
# ---------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------
# Evaluation Metrics
# ---------------------------------------------

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# ---------------------------------------------
# Confusion Matrix
# ---------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ---------------------------------------------
# ROC Curve
# ---------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")

plt.plot([0,1], [0,1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

# ---------------------------------------------
# Threshold Tuning
# ---------------------------------------------

threshold = 0.40

custom_prediction = (y_prob >= threshold).astype(int)

print("\nUsing Threshold =", threshold)

print("Precision:", precision_score(y_test, custom_prediction))
print("Recall   :", recall_score(y_test, custom_prediction))