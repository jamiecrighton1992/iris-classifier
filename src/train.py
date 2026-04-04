
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------
# Loading the data

iris = load_iris()

X = iris.data  # shape(150,4) 150 samples with 4 numerical features
y = iris.target  # shape(150,)

# ----------------------------------------
# Splitting into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    # 0.2 = 20% of data is for testing and 80% is for training/ random state makes the outcome determanistic(42 makes the outcome reproducable)
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------
# Initialise the model by importing a decision tree classifier
# The desicion tree classifier has the same random state as our test set

model = DecisionTreeClassifier(random_state=42)  # Importing the tree

# ----------------------------------------
# Training(fit) the model

# Any model in the library is trained with an .fit(X,y) call
model.fit(X_train, y_train)
# scikit-learn uses this to automatically find patterns in X_tain that corrispond to y_train (e.g if petal length is <2.5cm then the flower is setosa)

# ------------------------------------
# Making predictions

# the .predict() method takes X_test adn returns an array of predicted labels for each test sample
y_pred = model.predict(X_test)
# y_pred will have 30entries(20%/30 samples)
# each of these being 0,1,2(Setosa(0), Versicolor(1), Verginica(2)) *Scikit-learn Iris encoding

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
