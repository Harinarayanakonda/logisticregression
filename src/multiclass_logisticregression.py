import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target  # Class labels (0, 1, 2)

# Step 2: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Train a Multinomial Logistic Regression model (Softmax)
model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Test cases - Predict for custom data input
# Example input for a custom test case (replace with any values you want to test)
test_input = np.array([[13.2, 2.3, 2.3, 18.0, 101.0, 2.80, 3.12, 0.25, 1.45, 3.85, 0.63, 1.75, 830.0]])

# Predicting for the custom test input
test_prediction = model.predict(test_input)

# Output the prediction for custom input
print(f"\nTest Case Input: Alcohol=13.2, Malic acid=2.3, Ash=2.3, Alcalinity of ash=18.0, Magnesium=101.0, Total phenols=2.80, Flavanoids=3.12, Nonflavanoid phenols=0.25, Proanthocyanins=1.45, Color intensity=3.85, Hue=0.63, OD280/OD315=1.75, Proline=830.0")
print(f"Predicted class: {test_prediction[0]} (0: Class 0, 1: Class 1, 2: Class 2)")

# Additional test cases for all classes (Class 0, Class 1, and Class 2)
test_inputs = np.array([
    [13.0, 2.3, 2.4, 18.5, 105.0, 2.85, 3.20, 0.26, 1.50, 3.90, 0.64, 1.80, 840.0],  # Class 0
    [12.5, 1.9, 2.2, 16.8, 102.0, 2.55, 3.05, 0.24, 1.38, 3.75, 0.61, 1.70, 820.0],  # Class 1
    [13.5, 2.1, 2.5, 17.2, 106.0, 2.90, 3.15, 0.23, 1.42, 3.80, 0.62, 1.78, 830.0]   # Class 2
])

# Predicting for the test inputs (Class 0, 1, and 2)
test_predictions = model.predict(test_inputs)

# Output predictions for all classes
for idx, test in enumerate(test_inputs):
    print(f"\nTest Input {idx+1}: {test}")
    print(f"Predicted class: {test_predictions[idx]} (0: Class 0, 1: Class 1, 2: Class 2)")

