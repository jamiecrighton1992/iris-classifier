from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
