import json
from typing import Tuple, Dict, Iterable

import numpy as np
import torch
from geopy.geocoders import Nominatim
from pandas import read_csv, concat
from tqdm import tqdm
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.points = np.array(df.location)
        self.texts = [tokenizer(text, padding='max_length', max_length=32, truncation=True,
                                return_tensors="pt") for text in tqdm(df['text'])]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.texts[idx], self.points[idx]


class TwitterData:
    __slots__ = ("train_data", "val_data", "test_data")

    city_locations = "data/geocoding.json"
    train_users_data = "data/training_set_users.txt"
    test_users_data = "data/test_set_users.txt"
    train_tweets_data = "data/training_set_tweets.txt"
    test_tweets_data = "data/test_set_tweets.txt"
    train_saved_data = 'data/train_data.pt'
    val_saved_data = 'data/val_data.pt'
    test_saved_data = 'data/test_data.pt'

    def __init__(self, rebuild: bool = False, recode: bool = False, save: bool = True):

        if not rebuild:
            self.train_data = torch.load(TwitterData.train_saved_data)
            self.val_data = torch.load(TwitterData.val_saved_data)
            self.test_data = torch.load(TwitterData.test_saved_data)

        else:
            train_users = read_csv(TwitterData.train_users_data, sep='\t', names=['id', 'city'])
            test_users = read_csv(TwitterData.test_users_data, sep='\t', names=['id', 'location'])
            locations = TwitterData.geocode_us_cities(train_users.city.unique(), recode)
            train_users["location"] = train_users.city.map(locations)
            train_users = train_users.drop(columns='city')
            test_users.location = test_users.location.str.split(': ').apply(lambda x: x[1].split(',')).apply(
                lambda x: [float(x[0]), float(x[1])]
            )
            users = concat([train_users, test_users])
            users.location = users.location.apply(
                lambda x: None if x[0] < 25 or x[0] > 50 or x[1] > -65 or x[1] < -125 else x
            )
            users = users[users.location.notna()].drop_duplicates(subset='id').reset_index(drop=True)

            train_tweets = read_csv(TwitterData.train_tweets_data, sep='\t', names=['id', 'tweet_id', 'text', 'time'])
            test_tweets = read_csv(TwitterData.test_tweets_data, sep='\t', names=['id', 'tweet_id', 'text', 'time'])
            train_tweets.drop(columns=["tweet_id", "time"], inplace=True)
            test_tweets.drop(columns=["tweet_id", "time"], inplace=True)
            tweets = concat([train_tweets, test_tweets])
            tweets = tweets[tweets.text.notna() & tweets.id.notna()]
            df = tweets.set_index('id').join(users.set_index('id'), how='inner').reset_index(drop=True)

            df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=1),
                                                 [int(0.8 * len(df)), int(0.9 * len(df))])
            self.train_data, self.val_data, self.test_data = Dataset(df_train), Dataset(df_val), Dataset(df_test)
            if save:
                TwitterData.save(self.train_data, self.val_data, self.test_data)

    @staticmethod
    def save(train_data: Dataset, val_data: Dataset, test_data: Dataset) -> None:
        torch.save(train_data, TwitterData.train_saved_data)
        torch.save(val_data, TwitterData.val_saved_data)
        torch.save(test_data, TwitterData.test_saved_data)

    @staticmethod
    def geocode_us_cities(cities: Iterable[str], recode: bool = False) -> Dict[str, Tuple[float, float]]:
        if recode:
            locations = dict()
            for city in tqdm(cities):
                locations[city] = TwitterData.geocode(city)
            with open(TwitterData.city_locations, 'w') as f:
                json.dump(locations, f)
        else:
            with open(TwitterData.city_locations, 'r') as f:
                locations = json.load(f)
        return locations

    @staticmethod
    def geocode(address: str) -> Tuple[float, float]:
        geolocator = Nominatim(user_agent="twitter-geocoding")
        try:
            location = geolocator.geocode(address)
            return location.latitude, location.longitude
        except Exception:
            return None
