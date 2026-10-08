import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# =========================================================
# 1. TWO CLASS + LINEAR SVM
# Setosa(0) vs Versicolor(1)
# =========================================================

X2 = X[y != 2]
y2 = y[y != 2]

# Use only 2 features for plotting
X2 = X2[:, :2]

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="linear",C=1)
svm.fit(X_train, y_train)

print("Accuracy:", svm.score(X_test, y_test))
print("Support vectors:\n", svm.support_vectors_)
# =========================================================
# 2. TEST POINT NEAR MARGIN
# =========================================================

d = svm.decision_function(X_test)
i = np.argmin(abs(d))

test = X_test[i]
prediction = svm.predict([test])

print("\nNear-margin point:", test)
print("Decision value:", d[i])
print("Prediction:", prediction[0])

# =========================================================
# 3. PLOT SVM DECISION LINE
# =========================================================

w = svm.coef_[0]
b = svm.intercept_[0]

x = np.linspace(
    X_train[:, 0].min() - 1,
    X_train[:, 0].max() + 1,
    100
)

# Decision boundary: w1*x + w2*y + b = 0
line = -(w[0] * x + b) / w[1]

# Margins
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

plt.figure(figsize=(8, 6))

# Data points
plt.scatter(                                                                  
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    label="Training data"
)

# Decision line
plt.plot(
    x, line,
    "k-",
    label="Decision Boundary"
)

# Margins
plt.plot(
    x, margin1,
    "k--",
    label="Margin"
)

plt.plot(
    x, margin2,
    "k--"
)

# Support vectors
plt.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="green",
    label="Support Vectors"
)

# Test point
plt.scatter(
    test[0],
    test[1],
    marker="X",
    s=250,
    color="yellow",
    edgecolors="black",
    label="Test Point"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.grid()
plt.show()
# =========================================================
# 4. OVERLAPPING CLASSES
# Versicolor(1) vs Virginica(2)
# =========================================================

X3 = X[y != 0]
y3 = y[y != 0]

Xtr, Xte, ytr, yte = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

svm2 = SVC(kernel="linear", C=1)
svm2.fit(Xtr, ytr)

print("\nOverlapping classes accuracy:",
      svm2.score(Xte, yte))
# =========================================================
# 5. SINGLE CLASS TRAINING
# Only Setosa
# =========================================================

X_one = X[y == 0]

scaler = StandardScaler()
X_one = scaler.fit_transform(X_one)

ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.05
)

ocsvm.fit(X_one)

# Test all Iris samples
X_all = scaler.transform(X)

pred = ocsvm.predict(X_all)

print("\nOne-Class SVM:")
print("+1 = Inlier, -1 = Outlier")
print(pred)
