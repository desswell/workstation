import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import time

access_token = 'bb01e286bb01e286bb01e2860ab81977f2bbb01bb01e286dd647517a5bcfa4df0bccfcd'
version = '5.131'


def get_posts_with_keywords(owner_id, keywords, yaer_thread, month_thread):
    posts = []
    offset = 0
    count = 100
    pbar = tqdm(desc="Хаваю посты", unit="post")
    data_thread = datetime(yaer_thread, month_thread, 1).timestamp()

    while True:
        url = 'https://api.vk.com/method/wall.get'
        params = {
            'access_token': access_token,
            'v': version,
            'owner_id': owner_id,
            'count': count,
            'offset': offset
        }
        response = requests.get(url, params=params).json()

        if 'response' in response:
            items = response['response']['items']
            items = [item for item in items if item['date'] >= data_thread]
            if not items:
                break
            posts.extend(items)
            pbar.update(len(items))
            if len(items) < count:
                break
            offset += count
        else:
            print("Error:", response)
            break

    pbar.close()
    filtered_posts = [post for post in posts if any(keyword.lower() in post['text'].lower() for keyword in keywords)]
    return filtered_posts


def get_all_comments(owner_id, post_id):
    comments = []
    offset = 0
    count = 100

    while True:
        url = 'https://api.vk.com/method/wall.getComments'
        params = {
            'access_token': access_token,
            'v': version,
            'owner_id': owner_id,
            'post_id': post_id,
            'extended': 1,
            'fields': 'from_id, date, text',
            'count': count,
            'offset': offset,
            'thread_items_count': 10
        }
        response = requests.get(url, params=params).json()
        time.sleep(0.34)
        if 'response' in response:
            items = response['response']['items']
            comments.extend(items)
            if len(items) < count:
                break
            offset += count
        else:
            print("Error:", response)
            break

    all_comments = []
    for com in comments:
        all_comments.append((com['text'], datetime.fromtimestamp(com['date'])))
        if 'thread' in com and com['thread']['count'] > 0:
            thread_comments = com['thread']['items']
            for thread_com in thread_comments:
                all_comments.append((thread_com['text'], datetime.fromtimestamp(thread_com['date'])))

    return all_comments


def get_comments_for_keywords(owner_id, keywords, yaer_thread, month):
    posts = get_posts_with_keywords(owner_id, keywords, yaer_thread, month)
    df_comments = []
    for post in tqdm(posts, desc="Шарю комментарии по постам..."):
        post_id = post['id']
        comments = get_all_comments(owner_id, post_id)
        df_comments.extend(comments)

    return pd.DataFrame(df_comments, columns=["Comment", "Date"])

year_now = datetime.now().year
year = year_now
month_now = datetime.now().month
month = month_now - 1
owner_id = -40316705
keywords = ['']
df_comments = get_comments_for_keywords(owner_id, keywords, year, month)
print(df_comments)
