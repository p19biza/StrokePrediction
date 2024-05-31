import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# Load the preprocessed data
data = pd.read_csv('/Users/gpbiz/Desktop/Project/PreprocessedDatasetUnder.csv')

# Split the data into features and target       
X = data.drop('stroke', axis=1)
y = data['stroke']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a decision tree classifier
lr = LogisticRegression(random_state=42, C=1, solver='liblinear')

# Train the classifier on the training data
lr.fit(X_train_scaled, y_train)

# Extract feature importances
coefficients = np.abs(lr.coef_[0])
scaler = MinMaxScaler(feature_range=(1, 10))
scaled_coefficients = scaler.fit_transform(coefficients.reshape(-1, 1)).flatten()

# Create a dictionary of feature importances
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': scaled_coefficients})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Display the feature importances
print("Feature Importances (scaled from 1 to 10):")
for feature, importance in importance_df.items():
    print(f"{feature}: {importance}")

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importance - Logistic Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Perform a 10-fold cross-validation on the training data
scores = cross_val_score(lr, X_train_scaled, y_train, cv=10)

# Make predictions on the test data
y_pred = lr.predict(X_test_scaled)

# Calculate metrics
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Make probability predictions on the test data
y_pred_prob = lr.predict_proba(X_test_scaled)[:, 1]

# Calculate the AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print the metrics
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n {conf_matrix}")
print(f"Mean cross-validation score: {scores.mean()}")
print(f"ROC AUC: {roc_auc}")

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()