from catboost import CatBoostRegressor
from typing import List
from preprocessing import prepare_users, prepare_events, create_dataset, load_all, load_city_events
import time
import datetime
from tabulate import tabulate
import pandas as pd
import numpy as np
from numpy.random import choice, seed
import copy
from pprint import pprint
seed(10000)

evs, users = load_all()
city_events = load_city_events()


def predict(user_id: str, events_id: List[str]) -> List[str]:
    events_id = np.unique(events_id)
    cbr = CatBoostRegressor()
    cbr.load_model('model_dump')
    user = None
    for i in users:
        if i['id'] == user_id:
            user = i
    if user is None:
        raise RuntimeError(f'User with user_id: {user_id} does not exist')
    events = [event for event in evs if event['id'] in events_id]
    pairs = [(user['id'], event_id) for event_id in events_id]
    if 'ratedEvents' in user.keys():
        del user['ratedEvents']
    user = prepare_users([user])
    events = prepare_events(events)
    dataset = create_dataset(pairs, user, events, mode='test')
    dataset.drop_duplicates(subset=['event_id'], keep='last', inplace=True)
    dataset['score'] = cbr.predict(dataset)
    dataset = dataset.sort_values(by='score', ascending=False)
    return dataset['event_id'].values.tolist()


def get_sample_prediction(city: str):
    random_user = choice(users)
    max_events_num = 50
    response_events = []
    for _ in range(max_events_num):
        t = choice(city_events[city])
        response_events.append({'id': t['id'],
                                'eventTags': t['eventTags'],
                                'contentRating': t['contentRating'],
                                'city': city})
    response_events = pd.DataFrame(response_events)[['id', 'eventTags', 'contentRating', 'city']]
    random_user = pd.DataFrame([random_user])[['id', 'tags', 'ratedEvents']]
    response_events.drop_duplicates(subset=['id'], inplace=True)
    print('User:')
    print(tabulate(random_user, headers='keys', tablefmt='psql'))
    print('Raw events answer:')
    print(tabulate(response_events, headers='keys', tablefmt='psql'))
    print('\n'*3)
    prediction = predict(random_user['id'].values.tolist()[0], response_events['id'].values.tolist())
    raws = []
    for i in range(len(prediction)):
        raws.append(response_events[response_events['id'] == prediction[i]].index.values[0])

    response_events = response_events.reindex(raws)
    print(response_events.shape)
    print('Ranked events answer:')
    print(tabulate(response_events, headers='keys', tablefmt='psql'))


get_sample_prediction('moscow')
