import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt 
import seaborn as sns

# Load the data
data = pd.read_csv('/Users/gpbiz/Desktop/Project/healthcare-dataset-stroke-data.csv')

# Count the occurrences of each class in the 'stroke' column
stroke_counts_a = data['stroke'].value_counts()

# Plot the distribution of the 'stroke' column before preprocessing
plt.figure(figsize=(8, 6))
stroke_counts_a.plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of stroke cases before preprocessing')
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Stroke', 'Stroke'], rotation=0)
plt.grid(axis='y')

# Annotate the bars with the counts
for i in range(len(stroke_counts_a)):
    plt.text(i, stroke_counts_a[i] + 50, str(stroke_counts_a[i]), ha='center', va='bottom')

# Remove grid lines
plt.grid(False)
plt.show()

# Drop the 'id' column
data = data.drop('id', axis=1)

# Fill the missing values in the 'bmi' column with the mean value
mean_bmi = data['bmi'].mean()
data['bmi'] = data['bmi'].fillna(mean_bmi)

# Display rows where BMI is greater than 60
bmi_above_60 = data[data['bmi'] > 60]
print(f"\nRows where BMI is greater than 60:\n{bmi_above_60}")

# Identifying and handling outliers in the BMI column
Q1 = data['bmi'].quantile(0.25)
Q3 = data['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
data = data[(data['bmi'] >= lower_bound) & (data['bmi'] <= upper_bound)]

print(data.head())

#One-hot encode the categorical columns
#categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
#data = pd.get_dummies(data, columns=categorical_columns)

# Label encode the categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the data into features and target
X = data.drop('stroke', axis=1)
y = data['stroke']

# Print class distribution before undersampling
print("Class distribution before undersampling:")
print(y.value_counts())

# Undersampling
rus = RandomUnderSampler(random_state=1)
X_res, y_res = rus.fit_resample(X, y)

# Print class distribution after undersampling
print("\nClass distribution after undersampling:")
print(y_res.value_counts())

# Save the preprocessed data
new_data = pd.concat([X_res, y_res], axis=1)
new_data.to_csv('/Users/gpbiz/Desktop/Project/PreprocessedDatasetUnder.csv', index=False)

# Plot the distribution of the 'stroke' column after preprocessing
stroke_counts_b = y_res.value_counts()
plt.figure(figsize=(8, 6))
stroke_counts_b.plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of stroke cases after preprocessing')
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Stroke', 'Stroke'], rotation=0)
plt.grid(axis='y')

# Annotate the bars with the counts
for i in range(len(stroke_counts_b)):
    plt.text(i, stroke_counts_b[i], str(stroke_counts_b[i]), ha='center', va='bottom')

# Remove grid lines
plt.grid(False)
plt.show()

# Optional: Plot histograms and box plots for other features
def plot_features_distribution(df, features):
    num_features = len(features)
    fig, axes = plt.subplots(2, num_features, figsize=(5 * num_features, 10))
    
    for i, feature in enumerate(features):
        # Histogram
        sns.histplot(df[feature], kde=True, ax=axes[0, i])
        axes[0, i].set_title(f'Histogram of {feature}')
        
        # Box plot
        sns.boxplot(x=df[feature], ax=axes[1, i])
        axes[1, i].set_title(f'Boxplot of {feature}')
    
    plt.tight_layout()
    plt.show()

# List of features to plot
features = new_data.columns

# Group features in sets of three
for i in range(0, len(features), 3):
    group = features[i:i+3]
    plot_features_distribution(new_data, group)

pd.set_option('display.max_columns', None)

# Verifcation of one-hot encoding
print(new_data.head())