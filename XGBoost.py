import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Load the preprocessed data
data = pd.read_csv('/Users/gpbiz/Desktop/Project/PreprocessedDatasetUnder.csv')

# Split the data into features and target       
X = data.drop('stroke', axis=1)
y = data['stroke']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Create a decision tree classifier
xgb = XGBClassifier(random_state=42, max_depth=4)

# Train the classifier on the training data
xgb.fit(X_train, y_train)

# Calculate feature importances
importances = xgb.feature_importances_

# Normalize the importances to a range of 1 to 10
importances_normalized = 1 + 9 * (importances - np.min(importances)) / (np.max(importances) - np.min(importances))

# Get the indices of the features sorted by importance
indices = np.argsort(importances_normalized)[::-1]

# Print the feature ranking
print("Feature ranking (1-10):")

ranking=[]
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
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Perform a 10-fold cross-validation on the training data
scores = cross_val_score(xgb, X_train, y_train, cv=10)

# Make predictions on the test data
y_pred = xgb.predict(X_test)

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

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()