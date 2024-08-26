from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors

# Загрузка данных
df = pd.read_csv('structed_logs/u_ex240109_x.csv')

# Преобразование даты и времени в timestamp
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time']).astype(np.int64) // 10**9

# Подготовка данных для кластеризации: включаем timestamp, sc-status и cs-method
features = df[['timestamp', 'sc-status', 'cs-method', 'c-ip']]

# Преобразование категориальных данных метода запроса в числовые при помощи One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['timestamp', 'sc-status']),
        ('cat', OneHotEncoder(), ['cs-method', 'c-ip'])
    ])

X_processed = preprocessor.fit_transform(features)

# Применение HDBSCA
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, gen_min_span_tree=True)
clusters = clusterer.fit_predict(X_processed)

# Добавление меток кластера к исходным данным
df['cluster'] = clusters

# Поиск аномалий
anomalies = df[df['cluster'] == -1]

nn = NearestNeighbors(n_neighbors=1)

nn.fit(df[df['cluster'] != -1][['timestamp', 'sc-status']])

distances, indices = nn.kneighbors(anomalies[['timestamp', 'sc-status']])

anomalies['distance_to_nearest_cluster_point'] = distances
anomalies['probability'] = clusterer.probabilities_[df['cluster'] == -1]

secondary_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
anomalies_processed = StandardScaler().fit_transform(anomalies[['distance_to_nearest_cluster_point', 'probability']])
anomalies['secondary_cluster'] = secondary_clusterer.fit_predict(anomalies_processed)
filtered_anomalies = anomalies[anomalies['secondary_cluster'] == -1]
filtered_anomalies.to_csv('anomalies/filtered_anomalies2.csv', index=False)
