import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copulas.multivariate import GaussianMultivariate
from sklearn.preprocessing import LabelEncoder

# Load the Excel file
file_path = 'Датасет 1 (1).xlsx'
data = pd.read_excel(file_path)

# Exclude the ID column from the modeling process
data_to_model = data.drop(columns=['ID'])

# Encode categorical features
categorical_features = ['Цвет глаз', 'Любимая еда', 'Профессия']
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data_to_model[feature] = le.fit_transform(data_to_model[feature])
    label_encoders[feature] = le

# Fit the Gaussian copula model
model = GaussianMultivariate()
model.fit(data_to_model)

# Generate synthetic data
num_samples = len(data) * 3  # Generate twice the number of original samples
synthetic_data = model.sample(num_samples)

# Clip the synthetic data to the valid range of original labels
for feature in categorical_features:
    min_label = data_to_model[feature].min()
    max_label = data_to_model[feature].max()
    synthetic_data[feature] = synthetic_data[feature].clip(lower=min_label, upper=max_label)

# Decode categorical features
for feature in categorical_features:
    le = label_encoders[feature]
    synthetic_data[feature] = le.inverse_transform(synthetic_data[feature].round().astype(int))

# Round continuous features to match the precision of the original data
continuous_features = ['Высота', 'Вес', 'Количество глаз']
for feature in continuous_features:
    synthetic_data[feature] = synthetic_data[feature].round()

# Create a new ID column for synthetic data
synthetic_data['ID'] = range(len(data), len(data) + num_samples)

# Reorder columns to match the original data
synthetic_data = synthetic_data[['ID'] + list(data_to_model.columns)]

# Concatenate original data with synthetic data
combined_data = pd.concat([data, synthetic_data], ignore_index=True)

# Save the combined dataset to an Excel file
combined_data.to_excel('Датасет 2 (новый).xlsx', index=False)

# Plot correlation matrix before synthetic data generation
plt.figure(figsize=(10, 8))
sns.heatmap(data.drop(columns=['ID']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Before Synthetic Data Generation')
plt.show()

# Plot correlation matrix after synthetic data generation
plt.figure(figsize=(10, 8))
sns.heatmap(combined_data.drop(columns=['ID']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix After Synthetic Data Generation')
plt.show()

print(f"Original dataset shape: {data.shape}")
print(f"Synthetic dataset shape: {synthetic_data.shape}")
print(f"Combined dataset shape: {combined_data.shape}")

print("Combined dataset generated and saved as 'Датасет 1 (новый).xlsx'")
