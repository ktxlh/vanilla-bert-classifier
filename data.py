"""
Load news titles/text & labels
"""
import json
import random
from os import listdir
from os.path import isfile, join, isdir
from transformers import AutoModel, AutoTokenizer

import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
REAL, FAKE = 1, 0
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)


class NewsDataset(Dataset):
    def __init__(self, ids, features, labels):
        super(NewsDataset).__init__()
        self.ids = ids
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.features[idx], self.labels[idx]
    
def try_load_features(in_dir):
    if isfile(join(in_dir, "features-bert-baseline.npy")) and isfile(join(in_dir, "labels-bert-baseline.npy")):
        features = torch.from_numpy(np.load(join(in_dir, "features-bert-baseline.npy")))
        labels = torch.from_numpy(np.load(join(in_dir, "labels-bert-baseline.npy")))
        ids = [None for _ in range(len(labels))]
        return ids, features, labels
    else:
        return None, None, None

def preprocess_and_save(input, in_dir, max_seq_len, batch_size=256, model_name='bert-base-cased'):
    def standardize_(input):
        f = np.array([i[-1] for i in input])  # (num_instances, num_other_features)
        mean = np.mean(f, axis=0)
        std = np.std(f, axis=0)
        f = (f - mean) / std
        for i, ff in enumerate(f):
            input[i][-1] = ff.tolist()
    def detuple_embed(tokenizer, model, max_seq_len, batch_size, input):
        with torch.no_grad():
            ids = [i[0] for i in input]
            text = [i[1] for i in input]
            labels = [i[2] for i in input]
            other_features = [i[3] for i in input]
            features = []
            num_batches = (len(text) + batch_size - 1) // batch_size
            for i_batch in trange(num_batches, desc=f'embed'):
                t = text[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))]
                f = torch.tensor(
                    other_features[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))], 
                    dtype=torch.float32, requires_grad=False)
                inputs = tokenizer(t, return_tensors="pt", max_length=max_seq_len, padding='max_length', truncation=True)
                inputs = {k : v.to(device) for k, v in inputs.items()}
                pooled_output = model(**inputs).pooler_output.cpu()
                f = torch.tanh(f)
                feature = torch.cat([pooled_output, f], dim=-1)
                features.append(feature)
            features = torch.cat(features, dim=0)
        return ids, features, labels
    standardize_(input)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_len)
    model = AutoModel.from_pretrained(model_name, return_dict=True).to(device)
    random.shuffle(input)
    ids, features, labels = detuple_embed(tokenizer, model, max_seq_len, batch_size, input)
    np.save(join(in_dir, "features-bert-baseline.npy"), features.numpy())
    np.save(join(in_dir, "labels-bert-baseline.npy"), np.array(labels))
    return ids, features, labels

def split_input(ids, features, labels):
    n_train = int(len(labels) * 0.9 * 0.8)
    n_valid = n_train + int(len(labels) * 0.9 * 0.2)
    train = NewsDataset(ids[:n_train], features[:n_train], labels[:n_train])
    valid = NewsDataset(ids[n_train:n_valid], features[n_train:n_valid], labels[n_train:n_valid])
    test = NewsDataset(ids[n_valid:], features[n_valid:], labels[n_valid:])
    return train, valid, test

def read_pheme():
    in_dir = '/rwproject/kdd-db/20-rayw1/pheme-figshare'
    ids, features, labels = try_load_features(in_dir)
    if labels == None:
        rumorities = {'non-rumours': REAL, 'rumours': FAKE}
        input = []
        for event_raw in listdir(in_dir):
            if event_raw[-16:] != '-all-rnr-threads': continue
            # {event}-all-rnr-threads
            event = event_raw[:-16]
            for rumority, label in rumorities.items():
                for news_id in tqdm(listdir(join(in_dir, event_raw, rumority)), desc=f'pheme-{event}-{rumority}'):
                    if news_id == '.DS_Store': continue
                    tweets_dir = join(in_dir, event_raw, rumority, news_id, 'source-tweets')
                    for tweets_fn in listdir(tweets_dir):
                        if tweets_fn == '.DS_Store': continue 
                        with open(join(tweets_dir, tweets_fn), 'r') as f:
                            tweet = json.load(f)
                            other_features = [
                                tweet["favorite_count"], tweet["retweet_count"], tweet['user']['followers_count'], 
                                tweet['user']['statuses_count'], tweet['user']['friends_count'], tweet['user']['favourites_count'],
                                len(tweet['user']['description'].split(' ')) if tweet['user']['description'] else 0,
                            ]
                            input.append([tweet["id_str"], tweet['text'], label, other_features])
        ids, features, labels = preprocess_and_save(input, in_dir, max_seq_len=49)
    return split_input(ids, features, labels)

def read_politifact():
    in_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset/politifact'
    ids, features, labels = try_load_features(in_dir)
    if labels == None:
        rumorities = {'real': REAL, 'fake': FAKE}
        input = []
        no_content_news = []
        for rumority, label in rumorities.items():
            for news_id in tqdm(listdir(join(in_dir, rumority)), desc=f'politifact-{rumority}'):
                content_fn = join(in_dir, rumority, news_id, 'news content.json')
                if not isfile(content_fn):
                    no_content_news.append(news_id)
                    continue
                with open(content_fn, 'r') as f:
                    content = json.load(f)
                has_image = int(len(content["top_img"]) > 0)
                num_images = len(content["images"])
                num_exclam = (content["title"] + content["text"]).count("!")
                tp = join(in_dir, rumority, news_id, 'tweets')
                num_tweets = len(listdir(tp)) if isdir(tp) else 0
                rp = join(in_dir, rumority, news_id, 'retweets')
                num_retweets = len(listdir(rp)) if isdir(rp) else 0
                other_features = [has_image, num_images, num_exclam, num_tweets, num_retweets]
                input.append([news_id, content['title'] + " [SEP] " + content["text"], label, other_features])
        ids, features, labels = preprocess_and_save(input, in_dir, max_seq_len=490)
    return split_input(ids, features, labels)
        
