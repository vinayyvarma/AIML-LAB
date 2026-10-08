from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ==================================================
# TEST CASE 1: IRIS DATASET - K = 3
# ==================================================

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

iris_pred = model.predict(x_test)

print("\nTEST CASE 1: IRIS DATASET")
print("Input: Iris dataset, k=3")
print("Accuracy:", model.score(x_test, y_test))
print("Status: SUCCESS")


# ==================================================
# TEST CASE 2: NUMERIC DATA
# ==================================================

X2 = [[1,1], [2,2], [3,3],
      [10,10], [11,11], [12,12]]

y2 = [0, 0, 0, 1, 1, 1]

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X2, y2)

pred2 = model2.predict([[2,2]])

print("\nTEST CASE 2: NUMERIC DATA")
print("Input: [[2,2]]")
print("Output:", pred2)
print("Expected: Class 0")
print("Status: SUCCESS")


# ==================================================
# TEST CASE 3: CLEARLY SEPARATED CLASSES
# ==================================================

X3 = [[1,1], [1,2], [2,1], [2,2],
      [10,10], [10,11], [11,10], [11,11]]

y3 = [0, 0, 0, 0, 1, 1, 1, 1]

model3 = KNeighborsClassifier(n_neighbors=3)
model3.fit(X3, y3)

pred3 = model3.predict([[1.5,1.5], [10.5,10.5]])

print("\nTEST CASE 3: CLEARLY SEPARATED CLASSES")
print("Input: [[1.5,1.5], [10.5,10.5]]")
print("Output:", pred3)
print("Expected: [0 1]")
print("Status: SUCCESS")


# ==================================================
# TEST CASE 4: TIE AMONG NEIGHBORS
# ==================================================

X4 = [[0,0], [0,2], [2,0], [2,2]]
y4 = [0, 0, 1, 1]

model4 = KNeighborsClassifier(n_neighbors=4)
model4.fit(X4, y4)

pred4 = model4.predict([[1,1]])

print("\nTEST CASE 4: TIE AMONG NEIGHBORS")
print("Input: [[1,1]]")
print("Output:", pred4)
print("Expected: Tie handled consistently")
print("Status: EDGE CASE")


# ==================================================
# TEST CASE 5: K LARGER THAN DATASET
# ==================================================

X5 = [[1], [2], [3]]
y5 = [0, 1, 1]

print("\nTEST CASE 5: K LARGER THAN DATASET")

try:
    model5 = KNeighborsClassifier(n_neighbors=5)
    model5.fit(X5, y5)

    pred5 = model5.predict([[2]])

    print("Output:", pred5)

except ValueError:
    print("Input: 3 samples, k=5")
    print("Expected Output: Error")
    print("Actual Output: k is larger than training samples")
    print("Status: FAILURE / EDGE CASE")


# ==================================================
# FINAL IRIS RESULTS
# ==================================================

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, iris_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, iris_pred))
