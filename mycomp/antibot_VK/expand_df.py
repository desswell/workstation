import pandas as pd
import json
from tqdm import tqdm


def parse_and_expand(df):
    result = pd.DataFrame()

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        occupation_data = json.loads(row['occupation']) if pd.notna(row['occupation']) and row['occupation'] else {}
        personal_data = json.loads(row['personal']) if pd.notna(row['personal']) and row['personal'] else {}
        new_row = pd.Series()
        for key, value in occupation_data.items():
            new_row[f"occupation_{key}"] = value

        for key, value in personal_data.items():
            new_row[f"personal_{key}"] = value

        result = result._append(new_row, ignore_index=True)

    return result


df = pd.read_csv('shaman_focus_group.csv', sep=';')
expanded_df = parse_and_expand(df)
str = 'occupation_id,occupation_name,occupation_type,personal_langs,personal_alcohol,personal_smoking,personal_life_main,personal_langs_full,personal_inspired_by,personal_people_main,personal_political,occupation_city_id,occupation_country_id,occupation_graduate_year,personal_religion,personal_religion_id'
str = str.split(',')
for i in str:
    expanded_df[i] = expanded_df[i].fillna('-')
df = pd.concat([df, expanded_df], axis=1)

df.to_csv('expanded_df.csv', index=False)

print(expanded_df)
