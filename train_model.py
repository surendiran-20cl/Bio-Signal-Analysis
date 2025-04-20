# This script loads the data, preprocesses it, selects features, trains the model, and saves the scaler and best model.

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("smoking.csv")

# Drop ID column
df.drop(columns=['ID'], inplace=True)

# Convert 'Y'/'N' to 1/0 in categorical binary columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].replace({'Y': 1, 'N': 0})

# One-hot encode 'gender'
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Split features and target
X = df.drop('smoking', axis=1)
y = df['smoking']

# Feature selection using ExtraTreesClassifier
model_fs = ExtraTreesClassifier()
model_fs.fit(X, y)

# Select top 15 features
important_features = pd.Series(model_fs.feature_importances_, index=X.columns)
top_features = important_features.nlargest(15).index.tolist()
X = X[top_features]

# Save feature list for reference (optional)
with open("features_used.txt", "w") as f:
    f.write("\n".join(top_features))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Bagging Classifier
model = BaggingClassifier(estimator=ExtraTreesClassifier(), n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "bagging_model.pkl")
joblib.dump(scaler, "scaler.pkl")
