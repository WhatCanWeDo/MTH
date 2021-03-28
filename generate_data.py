import os
import requests
from numpy.random import choice, randint, seed, sample
from config import cities, tags
import pandas as pd

seed(42)


def get_data(city: str, max_data_len: int, tag: str = '', mode: str = 'events/actual'):
    _AFISHA_API_BASE = 'https://afisha.yandex.ru/api/'
    res = []
    res_len = 0
    full_url = os.path.join(_AFISHA_API_BASE, mode)
    while len(res) < max_data_len:
        response = requests.get(full_url, params={
            'limit': 20,
            'offset': str(res_len),
            'city': city,
            'tag': tag,
        }).json()

        res += [{'dateStarted': data['scheduleInfo']['dateStarted'],
                 'dateEnd': data['scheduleInfo']['dateEnd'],
                 'eventTags': [i['code'] for i in data['event']['tags']],
                 'contentRating': data['event']['contentRating'],
                 'userRating': data['event']['userRating'],
                 'id': data['event']['id'],
                 'coordinates': data['scheduleInfo']['onlyPlace']['coordinates']
                 if data['scheduleInfo']['onlyPlace'] is not None else None,
                 'address': data['scheduleInfo']['onlyPlace']['address']
                 if data['scheduleInfo']['onlyPlace'] is not None else None,
                 'image': data['event']['image'],
                 'title': data['event']['title'],
                 'description': [],
                 'city': city,
                 } for data in response['data']]
        if res_len == len(res):
            break
        res_len = len(res)

    return res[:max_data_len]


print('Generating events...')
events = []
tag_events = {tag: [] for tag in tags}
city_events = {city: [] for city in cities}
for city in cities:
    for tag in tags:
        resp = [i for i in get_data(city=city, max_data_len=200, tag=tag)]
        events += resp
        tag_events[tag] += resp
        city_events[city] += resp
events = [i for city in cities for tag in tags for i in get_data(city=city, max_data_len=200, tag=tag)]
print('Generating users...')


def get_random_event_by_tag(tag):
    event = choice(tag_events[tag])
    return event['id']


def gen_users(num_users: int, all_tags):
    res = []
    for i in range(num_users):
        user = {
            'id': 'user' + str(i),
            'age': randint(10, 60),
            'tags': {tag: sample() * 8 - 4 for tag in all_tags},
            'ratedEvents': [],
            'gender': randint(0, 1),
        }

        tgs = [(k, v) for k, v in user['tags'].items()]
        tgs.sort(key=lambda x: x[1], reverse=True)
        for _ in range(100):
            if sample() > 0.4:
                continue
            if sample() > 0.85:
                t = -1
                if sample() > 0.6:
                    t = 1
                tag = tgs[randint(len(tgs)-3, len(tgs) - 1)][0]
                user['ratedEvents'].append((get_random_event_by_tag(tag), t))
                user['tags'][tag] += t * 0.5
            else:
                t = 1
                if sample() > 0.6:
                    t = -1
                tag = tgs[randint(0, 4)][0]
                user['ratedEvents'].append((get_random_event_by_tag(tag), t))
                user['tags'][tag] += t * 0.5
        res.append(user)
    return res


min_len = min(len(v) for _, v in city_events.items())
city_events = {k: v[:min_len] for k, v in city_events.items()}
users = gen_users(num_users=5000, all_tags=tags)
pd.DataFrame(events).to_csv('events.csv', index=False)
pd.DataFrame(users).to_csv('users.csv', index=False)
pd.DataFrame.from_dict(city_events).to_csv('city_events.csv', index=False)
