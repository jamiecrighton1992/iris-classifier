
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
iris = load_iris()

X = iris.data  # shape(150,4)
y = iris.target  # shape(150,)

# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

# ----------------------------------------

confusion_matrix(y_test, y_pred)

print(iris.feature_names, iris.target_names)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
print("Accuracy:", accuracy)

(os.makedirs("outputs", exist_ok=True))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=iris.target_names)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax, cmap=plt.cm.Greens, colorbar=False)

ax.set_title(f"Iris Classifier (accuracy={accuracy:.3f})")

plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")

plt.close(fig)

# -------------------------------------------

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/decision_tree_iris.joblib")
