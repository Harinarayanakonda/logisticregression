# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Step 2: Load the Iris dataset (multi-class)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Step 3: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train a Logistic Regression model (multi-class)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Calculate VIF for each feature in the dataset

# Adding a constant to the features
X_with_const = add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# Step 8: Display VIF results
print("\nVariance Inflation Factors (VIF):\n", vif_data)

# Step 9: Test Case - Predict for custom data input
# We'll use an example input for a custom test case
test_input = np.array([[5.0, 3.4, 1.5, 0.2]])  # Example: sepal_length=5, sepal_width=3.4, petal_length=1.5, petal_width=0.2

# Step 10: Model prediction for the test case
prediction = model.predict(test_input)

# Step 11: Output prediction
print("\nTest Case Input: sepal_length=5.0, sepal_width=3.4, petal_length=1.5, petal_width=0.2")
print(f"Predicted class: {prediction[0]} (class {iris.target_names[prediction[0]]})")
