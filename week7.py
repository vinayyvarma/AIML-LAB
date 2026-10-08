                                                                                                                                                                                                     import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# ============================================================
# UPLOAD SALARY DATASET
# ============================================================

uploaded = files.upload()

dataset = pd.read_csv('salary_Data.csv')

print("========== SALARY DATASET ==========")
print(dataset.head())
print("\nNumber of rows:", len(dataset))

# ============================================================
# PREPARE DATA
# ============================================================

X = dataset[['YearsExperience']].values
y = dataset['Salary'].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1/3,
    random_state=0
)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print("Actual Salaries:")
print(y_test)

print("\nPredicted Salaries:")
print(y_pred)
# ============================================================
# TRAINING DATA GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    color='red',
    label='Training Data'
)

plt.plot(
    X_train,
    model2.predict(X_train),
    color='blue',
    label='Regression Line'
)

plt.title("Salary vs Years of Experience - Training Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()


# ============================================================
# TEST DATA GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    color='red',
    label='Actual Test Data'
)

plt.plot(
    X_test,
    y_pred,
    color='blue',
    label='Predicted Regression Line'
)

plt.title("Salary vs Years of Experience - Test Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
