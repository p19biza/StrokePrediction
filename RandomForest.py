import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
import graphviz
import numpy as np
import pickle

# Load the preprocessed data
data = pd.read_csv('/Users/gpbiz/Desktop/Project/PreprocessedDatasetUnder.csv')

# Split the data into features and target       
X = data.drop('stroke', axis=1)
y = data['stroke']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Create a decision tree classifier
rf = RandomForestClassifier(random_state=42, n_estimators= 120, criterion='entropy', max_depth= 6)

# Train the classifier on the training data
rf.fit(X_train, y_train)

# Calculate feature importances
importances = rf.feature_importances_

# Normalize the importances to a range of 1 to 10
importances_normalized = 1 + 9 * (importances - np.min(importances)) / (np.max(importances) - np.min(importances))

# Get the indices of the features sorted by importance
indices = np.argsort(importances_normalized)[::-1]

# Print the feature ranking
print("Feature ranking (1-10):")

ranking = []
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances_normalized[indices[f]]}): {X_train.columns[indices[f]]}")
    ranking.append((X_train.columns[indices[f]], importances_normalized[indices[f]]))

# Convert the ranking to a DataFrame
ranking_df = pd.DataFrame(ranking, columns=['feature', 'importance'])

# Sort the DataFrame by importance
ranking_df = ranking_df.sort_values('importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=ranking_df)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Perform a 10-fold cross-validation on the training data
scores = cross_val_score(rf, X_train, y_train, cv=10)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Calculate metrics
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the metrics
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n {conf_matrix}")
print(f"Mean cross-validation score: {scores.mean()}")

# Get the first tree from the forest
estimator = rf.estimators_[0]

# Define feature_names and class_names
feature_names = X.columns
class_names = [str(x) for x in y.unique()]

dot_data = tree.export_graphviz(estimator, out_file=None, 
                                feature_names=feature_names,  
                                class_names=class_names,
                                filled=True)

# Graph
graph = graphviz.Source(dot_data, format="png") 
graph.render("random_forest_graphivz")


# Confusion matrix heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
with open('stroke_model.pkl', 'wb') as f:
    pickle.dump(rf, f)