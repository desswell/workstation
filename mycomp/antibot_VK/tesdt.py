import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('shaman_focus_group.csv', delimiter=';', encoding='utf-8')

data.replace("", np.nan, inplace=True)

data['last_seen'] = pd.to_datetime(data['last_seen'])

data.fillna(data.mean(numeric_only=True), inplace=True)

categorical_columns = data.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col].astype(str))
correlation_matrix = data.corr()

plt.figure(figsize=(40, 40))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
