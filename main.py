from catboost import CatBoostRegressor
from typing import List
from preprocessing import prepare_users, prepare_events, create_dataset, load_all, load_city_events
import time
import datetime
from tabulate import tabulate
import pandas as pd
from numpy.random import choice, seed
from pprint import pprint
seed(42)

evs, users = load_all()
city_events = load_city_events()


def predict(user_id: str, events_id: List[str]) -> List[str]:
    cbr = CatBoostRegressor()
    cbr.load_model('model_dump')
    user = [user for user in users if user['id'] == user_id]
    pprint(user)
    user = user[0]
    user['requestTime'] = time.mktime(datetime.datetime.strptime(user['requestTime'], "%Y-%m-%d").timetuple())
    events = [event for event in evs if event['id'] in events_id]
    pairs = [(user['id'], event['id']) for event in events]
    if 'ratedEvents' in user.keys():
        del user['ratedEvents']
    user = prepare_users([user])
    events = prepare_events(events)
    dataset = create_dataset(pairs, user, events, mode='test')
    dataset['score'] = cbr.predict(dataset)
    dataset = dataset.sort_values(by='score', ascending=False)
    return dataset['event_id'].values.tolist()


def get_sample_prediction(city: str, request_time: str):
    request_time = time.mktime(datetime.datetime.strptime(request_time, "%Y-%m-%d").timetuple())
    random_user = choice(users)
    random_user['requestTime'] = request_time
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
    print('User:')
    print(tabulate(random_user, headers='keys', tablefmt='psql'))
    print('Raw events answer:')
    print(tabulate(response_events, headers='keys', tablefmt='psql'))
    print('\n'*3)
    prediction = predict(random_user['id'], response_events['id'].values.tolist())
    response_events = response_events.reindex(prediction)
    print('Ranked events answer:')
    print(tabulate(response_events, headers='keys', tablefmt='psql'))


get_sample_prediction('moscow', '2021-03-28')