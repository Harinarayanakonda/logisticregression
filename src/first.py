import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

# 1. Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. Predictions and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 5. Accuracy and evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("🔍 Accuracy on Test Set:", round(accuracy, 4))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("🧮 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 6. Predict on a new sample
# Pick one sample from the test set
sample_index = 5
new_sample = X_test[sample_index].reshape(1, -1)
true_class = y_test[sample_index]
predicted_class = model.predict(new_sample)[0]
probabilities = model.predict_proba(new_sample)[0]

print(f"\n🧪 Predicting Test Sample #{sample_index}:")
print(f"→ True Class: {target_names[true_class]}")
print(f"→ Predicted Class: {target_names[predicted_class]}")
print(f"→ Class Probabilities: {dict(zip(target_names, np.round(probabilities, 4)))}")

# 7. Optional: Plot 2D decision boundary using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

model_2d = LogisticRegression()
model_2d.fit(X_reduced, y)

# Meshgrid for plotting decision boundary
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("Logistic Regression Decision Boundary (PCA 2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
