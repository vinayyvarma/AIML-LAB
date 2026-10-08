from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def prepare_data(data):
    df = pd.DataFrame(data)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    encoders = {}

    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    return X, y, y_encoder

play_tennis = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"]
]

from sklearn.tree import export_text

def show_tree(model, feature_names, test_case):
    print("\n" + "="*50)
    print(test_case)
    print("="*50)

    print(export_text(model, feature_names=feature_names))

    print("Tree Depth :", model.get_depth())
    print("No. of Leaves :", model.get_n_leaves())
    print()

testcase1:

X, y, encoder = prepare_data(play_tennis)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

show_tree(
    model,
    ["Outlook", "Temperature", "Humidity", "Wind"],
    "TEST CASE 1 : Play Tennis Dataset")

testcase2:

clear_data = [
    ["Red", "Yes"],
    ["Red", "Yes"],
    ["Blue", "No"],
    ["Blue", "No"]
]
X, y, encoder = prepare_data(clear_data)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

show_tree(
    model,
    ["Color"],
    "TEST CASE 2 : Clear Class Separation"
)
testcase3:
mixed_data = [
    ["Young", "High", "Yes"],
    ["Young", "Low", "No"],
    ["Middle", "High", "Yes"],
    ["Middle", "Low", "Yes"],
    ["Old", "High", "Yes"],
    ["Old", "Low", "No"]
]
X, y, encoder = prepare_data(mixed_data)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

show_tree(
    model,
    ["Age", "Income"],
    "TEST CASE 3 : Mixed Attributes"
)

testcase4:

same_class = [
    ["A", "Yes"],
    ["B", "Yes"],
    ["C", "Yes"],
    ["D", "Yes"],
    ["E", "Yes"]
]
X, y, encoder = prepare_data(same_class)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

show_tree(
    model,
    ["Attribute"],
    "TEST CASE 4 : All Samples Same Class"
)

testcase5:

missing_data = [ ["Sunny", "Hot", "Yes"],
                 ["Rain", None, "No"],
                 ["Overcast", "Mild", "Yes"],
                [None, "Cool", "Yes"],
                 ["Sunny", "Cool", "No"]
]
X, y, encoder = prepare_data(missing_data)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)
show_tree( model, ["Weather", "Temperature"],
          "TEST CASE 5 : Missing Attribute Value"
)
