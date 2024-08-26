# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# df = pd.read_csv('Pokemon.csv')
# df.drop(columns=['type2', 'generation', 'speed', 'Имя', 'ID', 'Тип'], inplace=True)
#
# encoder = LabelEncoder()
# df['type1_encoded'] = encoder.fit_transform(df['Легендарность'])
# df['Легендарность'] = df['type1_encoded']
# df.drop(columns=['type1_encoded'], inplace=True)
# df['ID'] = range(1, len(df) + 1)
# cols = ['ID'] + [col for col in df if col != 'ID']
# df = df[cols]
# # Randomly sample 100 rows for each class of 'Легендарность'
# df_sampled_0 = df[df['Легендарность'] == 0].sample(150, random_state=42)
# df_sampled_1 = df[df['Легендарность'] == 1].sample(50, random_state=42)
#
# # Concatenate the samples to form the final dataset
# df_final = pd.concat([df_sampled_0, df_sampled_1]).reset_index(drop=True)
#
# # Check the distribution in the final dataset
# df_final['Легендарность'].value_counts()
# # Shuffle the final dataframe
# df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Update the ID column to reflect the new order
# df_final['ID'] = df_final.index + 1
# df_remaining = df.drop(df_final.index)
#
# df_final_10 = df_remaining.sample(10, random_state=42).reset_index(drop=True)
#
# # Set the 'Легендарность' column to NaN (empty values)
# df_final_10['Легендарность'] = None
# df_final_10['ID'] = df_final_10.index + 1
# df_final.to_excel('Pokemon.xlsx', index=False)
# df_final_10.to_excel('Pokemon valid.xlsx', index=False)
# import pandas as pd
#
# file_path = 'Pokemon.xlsx'
# dataset = pd.read_excel(file_path)
# df = dataset.copy()
# from sklearn.preprocessing import MinMaxScaler
#
# features = ['CP', 'ОЗ', 'Атака', 'Защита', 'Ск_Атаки', 'Ск_Защиты', 'Легендарность']
#
# scaler = MinMaxScaler()
# df[features] = scaler.fit_transform(df[features])
#
# df['Рейтинг силы'] = (
#     0.4 * df['CP'] +  # CP is heavily weighted
#     0.2 * df['ОЗ'] +  # ОЗ is also important
#     0.1 * (df['Атака'] + df['Защита'] + df['Ск_Атаки'] + df['Ск_Защиты']) +  # Attack and defense stats are averaged
#     0.2 * df['Легендарность']  # Legendary status significantly boosts the rating
# )
#
# df['Рейтинг силы'] = df['Рейтинг силы'] * 9 + 1
# df['Рейтинг силы'] = df['Рейтинг силы'].round().astype(int)
# dataset['Рейтинг силы'] = df['Рейтинг силы']
# df.drop(columns=['Легендарность'])
# dataset.to_excel('Pokemon 2.xlsx', index=False)
import pandas as pd
df_pokemon2 = pd.read_excel('Pokemon 2.xlsx')
df_pokemon2.drop(columns='Легендарность', inplace=True)
# Randomly sample 200 rows from the original dataset (df_new)
df_random_200 = df_pokemon2.sample(200, random_state=42).reset_index(drop=True)
df_random_200['ID'] = df_random_200.index + 1
# Randomly sample 20 rows from the new dataset (df_pokemon2)
df_random_20 = df_pokemon2.sample(20, random_state=41).reset_index(drop=True)
df_random_20['ID'] = df_random_20.index + 1
df_random_20['Рейтинг силы'] = None

df_random_200.to_excel('Pokemon 2 (s).xlsx', index=False)
df_random_20.to_excel('Pokemon 2 valid.xlsx', index=False)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
#
# # Загрузка датасета
# df = pd.read_excel('Pokemon 2.xlsx')
#
# # Удаление существующей колонки 'Рейтинг силы' (если она есть)
# if 'Рейтинг силы' in df.columns:
#     df = df.drop(columns=['Рейтинг силы'])
#
#
# # Функция для добавления шума к данным
# def add_input_lag(df, noise_level=0.05):
#     df_noisy = df.copy()
#
#     # Добавляем случайный шум в пределах noise_level
#     for column in ['Сила', 'ОЗ', 'Атака', 'Защита', 'Скорость атаки', 'Скорость защиты']:
#         noise = np.random.uniform(-noise_level, noise_level, size=df_noisy.shape[0])
#         df_noisy[column] = df_noisy[column] * (1 + noise)
#
#     return df_noisy
#
#
# # Применение шума к датасету
# df_noisy = add_input_lag(df, noise_level=0.01)
#
# # Определение признаков и целевой переменной (сгенерированной случайно)
# features = ['Сила', 'ОЗ', 'Атака', 'Защита', 'Скорость атаки', 'Скорость защиты']
# target = np.random.randint(1, 11, size=df_noisy.shape[0])  # Случайно сгенерированный "Рейтинг силы"
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(df_noisy[features], target, test_size=0.2, random_state=42)
#
# # Обучение модели дерева решений
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)
#
# # Предсказание "Рейтинга силы" для всего датасета
# df_noisy['Рейтинг силы'] = model.predict(df_noisy[features])
#
# # Округление предсказаний до целых чисел и приведение их к диапазону от 1 до 10
# df_noisy['Рейтинг силы'] = df_noisy['Рейтинг силы'].round().astype(int).clip(1, 10)
#
# # Просмотр первых нескольких строк с новыми данными
# print(df_noisy.head())
#
# df['Рейтинг силы'] = df_noisy['Рейтинг силы']
#
# df.to_excel('Pokemon 2.xlsx', index=False)
