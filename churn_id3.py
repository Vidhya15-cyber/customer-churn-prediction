import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load data manually into DataFrame
data = pd.DataFrame({
    'ID': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
    'Contract Type': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month',
                      'Two year', 'One year', 'Month-to-month', 'Two year', 'One year'],
    'Support Calls': ['High', 'Low', 'High', 'Medium', 'Low',
                      'Low', 'High', 'Medium', 'Low', 'Low'],
    'Monthly Charges': ['High', 'Low', 'Medium', 'Medium', 'High',
                        'Low', 'High', 'Medium', 'Low', 'Medium'],
    'Internet Service': ['Yes', 'No', 'Yes', 'No', 'Yes',
                         'No', 'Yes', 'Yes', 'No', 'No'],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes',
              'No', 'Yes', 'Yes', 'No', 'No']
})

print("Data:\n", data)

# 2️⃣ Calculate Entropy of target variable (Churn)
def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -sum(p * np.log2(p) for p in probs)

target_entropy = entropy(data['Churn'])
print("\nEntropy of Churn:", round(target_entropy, 4))

# 3️⃣ Encode categorical variables
le = LabelEncoder()
for col in ['Contract Type', 'Support Calls', 'Monthly Charges', 'Internet Service', 'Churn']:
    data[col] = le.fit_transform(data[col])

# 4️⃣ Compute Information Gain for each feature
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / sum(counts)) * entropy(df[df[feature] == values[i]][target])
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy

features = ['Contract Type', 'Support Calls', 'Monthly Charges', 'Internet Service']
for f in features:
    ig = information_gain(data, f, 'Churn')
    print(f"Information Gain for {f}: {round(ig, 4)}")

# 5️⃣ Build Decision Tree
X = data[features]
y = data['Churn']

clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf.fit(X, y)

# 6️⃣ Show textual tree
tree_rules = export_text(clf, feature_names=features)
print("\nDecision Tree:\n", tree_rules)

# 7️⃣ Plot tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=features, class_names=['No','Yes'], filled=True)
plt.show()
