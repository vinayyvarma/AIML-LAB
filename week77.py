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


# ============================================================
# TEST CASE 1
# SUCCESS - LINEAR DATA POINTS
# Best-fit line should be generated
# ============================================================

print("\n")
print("==============================================")
print("TEST CASE 1: LINEAR DATA POINTS")
print("==============================================")

model1 = LinearRegression()
model1.fit(X, y)

slope1 = model1.coef_[0]
intercept1 = model1.intercept_

print("Slope:", slope1)
print("Intercept:", intercept1)

print("\nRegression Equation:")
print("Salary =", slope1, "* YearsExperience +", intercept1)

# Check whether model generated a line
if slope1 != 0:
    print("\nRESULT: PASS")
    print("Best-fit regression line generated successfully.")
else:
    print("\nRESULT: FAIL")


# ============================================================
# TEST CASE 2
# SUCCESS - TRAIN TEST DATASET
# Prediction should be close to expected salary
# ============================================================

print("\n")
print("==============================================")
print("TEST CASE 2: TRAIN-TEST DATASET")
print("==============================================")

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

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Display actual vs predicted
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test, y_pred):
    print(
        "Actual =", actual,
        " Predicted =", round(predicted, 2)
    )

# Check prediction quality
if r2 > 0.5:
    print("\nRESULT: PASS")
    print("Predictions are reasonably close to actual salaries.")
else:
    print("\nRESULT: FAIL")


# ============================================================
# TEST CASE 3
# SUCCESS - POSITIVE CORRELATION DATA
# Salary should increase with experience
# Positive slope should be obtained
# ============================================================

print("\n")
print("==============================================")
print("TEST CASE 3: POSITIVE CORRELATION")
print("==============================================")

model3 = LinearRegression()
model3.fit(X, y)

slope3 = model3.coef_[0]

print("Slope:", slope3)

if slope3 > 0:
    print("\nRESULT: PASS")
    print("Positive slope obtained.")
    print("Salary increases as YearsExperience increases.")
else:
    print("\nRESULT: FAIL")
    print("Negative or zero slope obtained.")


# ============================================================
# TEST CASE 4
# FAILURE / EDGE CASE
# CONSTANT TARGET VALUES
#
# We use the salary dataset but replace salary values
# with the mean salary.
# ============================================================

print("\n")
print("==============================================")
print("TEST CASE 4: CONSTANT TARGET VALUES")
print("==============================================")

# Keep YearsExperience unchanged
X4 = X.copy()

# Make every salary equal to the average salary
constant_salary = np.mean(y)

y4 = np.full(len(y), constant_salary)

model4 = LinearRegression()
model4.fit(X4, y4)

slope4 = model4.coef_[0]
intercept4 = model4.intercept_

print("Original average salary:", constant_salary)
print("Slope:", slope4)
print("Intercept:", intercept4)

if abs(slope4) < 0.000001:
    print("\nRESULT: PASS")
    print("Zero slope obtained for constant target values.")
else:
    print("\nRESULT: FAIL")


# ============================================================
# TEST CASE 5
# FAILURE / EDGE CASE
# OUTLIER-HEAVY SALARY DATA
#
# Add an extreme salary value to the dataset.
# ============================================================

print("\n")
print("==============================================")
print("TEST CASE 5: OUTLIER-HEAVY DATA")
print("==============================================")

# Copy original salary values
y5 = y.copy()

# Find the highest salary and make it an extreme outlier
highest_salary_index = np.argmax(y5)

original_salary = y5[highest_salary_index]

# Make salary 5 times larger
y5[highest_salary_index] = original_salary * 5

print("Original highest salary:", original_salary)
print("Outlier salary:", y5[highest_salary_index])

# Model without outlier
normal_model = LinearRegression()
normal_model.fit(X, y)

normal_slope = normal_model.coef_[0]

# Model with outlier
outlier_model = LinearRegression()
outlier_model.fit(X, y5)

outlier_slope = outlier_model.coef_[0]

print("\nSlope before outlier:", normal_slope)
print("Slope after outlier:", outlier_slope)

# Calculate predictions
normal_prediction = normal_model.predict(X)
outlier_prediction = outlier_model.predict(X)

normal_mse = mean_squared_error(y, normal_prediction)

# Compare original salaries with model trained on outlier data
outlier_mse = mean_squared_error(y, outlier_prediction)

print("\nMSE before outlier:", normal_mse)
print("MSE after outlier:", outlier_mse)

if outlier_mse > normal_mse:
    print("\nRESULT: PASS")
    print("Model error increased because of the outlier.")
else:
    print("\nRESULT: OUTLIER DID NOT INCREASE ERROR")


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


# ============================================================
# FINAL TEST SUMMARY
# ============================================================

print("\n")
print("================================================")
print("             TEST CASE SUMMARY")
print("================================================")

print("Test Case 1 : Linear Data          -> Best-fit line")
print("Test Case 2 : Train-Test Data      -> Prediction accuracy")
print("Test Case 3 : Positive Correlation -> Positive slope")
print("Test Case 4 : Constant Target      -> Zero slope")
print("Test Case 5 : Outlier Data         -> Error increases")
print("================================================")
