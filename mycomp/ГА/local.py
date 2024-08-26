import pandas as pd
from geopy.distance import geodesic

# Загрузить данные о расстояниях между городами из МКР
mkr_df = pd.read_excel('МКР.xlsx')

# Загрузить данные о городах и их координатах из файла .csv
cities_df = pd.read_csv('city.csv')

# Найдем координаты Великого Устюга
velikiy_ustyug_coords = cities_df.loc[cities_df['city'] == 'Великий Устюг', ['geo_lat', 'geo_lon']].values.flatten()
new_row = []

city = mkr_df.iloc[:, 0].values
for c in city:
    coords = cities_df.loc[cities_df['city'] == c, ['geo_lat', 'geo_lon']].values.flatten()[:2]
    print(c)
    distance = round(geodesic((velikiy_ustyug_coords), (coords)).kilometers)
    new_row.append(distance)
mkr_df['Великий Устюг'] = new_row
new_row = ['Великий Устюг'] + new_row + ['']
new_row = pd.Series(new_row, index=mkr_df.columns)
mkr_df = mkr_df._append(new_row, ignore_index=True)
mkr_df.to_excel('МКР.xlsx', index=False)
