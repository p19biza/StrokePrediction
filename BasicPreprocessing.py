import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Load the data
data = pd.read_csv('/Users/gpbiz/Desktop/Project/healthcare-dataset-stroke-data.csv')

# Drop the 'id' column
data = data.drop('id', axis=1)

# Fill the missing values in the 'bmi' column with the mean value
mean_bmi = data['bmi'].mean()
data['bmi'] = data['bmi'].fillna(mean_bmi)

# Label encode the categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoder = LabelEncoder()

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split the data into features and target
X = data.drop('stroke', axis=1)
y = data['stroke']

print(data['stroke'].value_counts())

# Save the data to a CSV file
data.to_csv('/Users/gpbiz/Desktop/Project/BasicPreprocessedDataset.csv', index=False)
