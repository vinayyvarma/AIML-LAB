# ID3 Decision Tree Classification using Entropy

# Step 1: Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Step 2: Create the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes',
                   'No', 'Yes', 'No', 'Yes', 'Yes',
                   'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# Step 3: Convert categorical data into numerical values
encoder = LabelEncoder()

for column in df.columns:
    df[column] = encoder.fit_transform(df[column])


print("\nEncoded Dataset:")
print(df)


# Step 4: Separate features and target
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']


# Step 5: Create ID3 Decision Tree using Entropy
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)


# Step 6: Train the model
model.fit(X, y)


# Step 7: Display the generated decision tree
print("\nDecision Tree:")
tree = export_text(
    model,
    feature_names=list(X.columns)
)

print(tree)


# Step 8: Display tree information
print("Tree Depth:", model.get_depth())
print("Number of Leaf Nodes:", model.get_n_leaves())


# Step 9: Make a prediction
sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPrediction for first sample:")
print("Predicted Class:", prediction[0])


# Step 10: Program completed
print("\nID3 Decision Tree program completed successfully.")

# this code for Graph Representation
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree - Play Tennis")
plt.show()
