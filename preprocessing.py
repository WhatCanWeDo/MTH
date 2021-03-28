import pandas as pd
import ast
import time
import datetime
from config import tags_dict, tags, cities
from pprint import pprint


def decode(filename, tags_to_encode):
    data = pd.read_csv(filename)
    for col in data.columns:
        data[col].fillna("0", inplace=True)
    data = data.to_dict('records')
    for item in data:
        for k, v in item.items():
            if k in tags_to_encode and isinstance(v, str):
                item[k] = ast.literal_eval(v)
    return data


def load_all():
    evs = decode('events.csv', ['eventTags', 'image', 'userRating', 'description', 'coordinates'])
    users = decode('users.csv', ['ratedEvents', 'age', 'gender', 'tags', 'queryTime'])
    return evs, users


def load_city_events():
    res = pd.read_csv('city_events.csv')
    res = res.to_dict('list')
    for k, _ in res.items():
        for i in range(len(res[k])):
            res[k][i] = ast.literal_eval(res[k][i])

    return res


def one_hot_dict(data, dict_tag: str, pref_to_add: str):
    for item in data:
        for k, v in item[dict_tag].items():
            item[pref_to_add+k] = v
        del item[dict_tag]
    return data


def prepare_events(events):
    keys_to_delete = ['image', 'address', 'description', 'title']
    for i in range(len(events)):
        for key_to_delete in keys_to_delete:
            if key_to_delete in events[i]:
                del events[i][key_to_delete]
        backup = events[i]['userRating']
        events[i]['userRating'] = backup['overall']['value']
        events[i]['usersRated'] = backup['overall']['count']
        event_tags = {tag: 0 for tag in tags}
        for tag in events[i]['eventTags']:
            for main_tag, ls in tags_dict.items():
                if tag in ls:
                    event_tags[main_tag] += 1
        events[i]['tags'] = event_tags
        del events[i]['dateStarted']
        del events[i]['dateEnd']
        del events[i]['city']
        del events[i]['eventTags']
        del events[i]['coordinates']
        events[i]['contentRating'] = int(events[i]['contentRating'].replace('+', ''))
    events = pd.DataFrame(one_hot_dict(events, 'tags', 'eventTag'))
    return events


def prepare_users(users_list):
    users_list = pd.DataFrame(one_hot_dict(users_list, 'tags', 'userTag_'))
    return users_list


def create_dataset(pairs, users_, events, mode='train'):
    cols = ['user_id', 'event_id']
    if mode == 'train':
        cols.append('score')
    dataset = pd.DataFrame(pairs, columns=cols)
    for data, id_col in [(users_, 'user_id'), (events, 'event_id')]:
        dataset = dataset.merge(data, how='left', left_on=id_col, right_on='id')
        dataset.drop(columns=['id'], inplace=True)
    return dataset


def prepare_for_training():
    evs, users = load_all()
    pairs = []
    for user in users:
        pairs += [(user['id'], event_id, score) for event_id, score in user['ratedEvents']]
        del user['ratedEvents']

    users = prepare_users(users)
    evs = prepare_events(evs)
    dataset = create_dataset(pairs, users, evs)
    target = dataset['score']
    dataset.drop(columns=['score'], inplace=True)
    return dataset, target
