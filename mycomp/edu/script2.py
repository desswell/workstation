import pandas as pd

df = pd.read_excel('new_data.xlsx')
df.drop(columns=['произведенное_исцеление'])
df.to_excel('new_data.xlsx', index=False)
