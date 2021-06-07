"""
Load news titles/text & labels
"""
import pandas as pd
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
    if isfile(join(in_dir, "features-bert-baseline.npy")) \
        and isfile(join(in_dir, "labels-bert-baseline.npy")) \
            and isfile(join(in_dir, "ids-bert-baseline.txt")):
        features = torch.from_numpy(np.load(join(in_dir, "features-bert-baseline.npy")))
        labels = torch.from_numpy(np.load(join(in_dir, "labels-bert-baseline.npy")))
        with open(join(in_dir, "ids-bert-baseline.txt"), 'r') as f:
            ids = [l.strip() for l in f.readlines()]
        return ids, features, labels
    else:
        return None, None, None

def preprocess_and_save(inpt, in_dir, max_seq_len, batch_size=256, model_name='bert-base-cased', transform='tanh'):
    def transform_(inpt):
        f = np.array([i[-1] for i in inpt])
        if transform == "standardize":
            mean = np.mean(f, axis=0)
            std = np.std(f, axis=0)
            f = (f - mean) / std
        elif transform == "tanh":
            f = np.tanh(f)
        for i, ff in enumerate(f):
            inpt[i][-1] = ff.tolist()
    def detuple_embed(tokenizer, model, max_seq_len, batch_size, inpt):
        with torch.no_grad():
            ids = [i[0] for i in inpt]
            text = [i[1] for i in inpt]
            labels = [i[2] for i in inpt]
            other_features = [i[3] for i in inpt]
            features = []
            num_batches = (len(text) + batch_size - 1) // batch_size
            for i_batch in trange(num_batches, desc=f'embed'):
                t = text[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))]
                f = torch.tensor(
                    other_features[i_batch * batch_size : min((i_batch + 1) * batch_size, len(text))], 
                    dtype=torch.float32, requires_grad=False)
                inpts = tokenizer(t, return_tensors="pt", max_length=max_seq_len, padding='max_length', truncation=True)
                inpts = {k : v.to(device) for k, v in inpts.items()}
                pooled_output = model(**inpts).pooler_output.cpu()
                f = torch.tanh(f)
                feature = torch.cat([pooled_output, f], dim=-1)
                features.append(feature)
            features = torch.cat(features, dim=0)
        return ids, features, labels
    transform_(inpt)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_len)
    model = AutoModel.from_pretrained(model_name, return_dict=True).to(device)
    random.shuffle(inpt)
    ids, features, labels = detuple_embed(tokenizer, model, max_seq_len, batch_size, inpt)
    np.save(join(in_dir, "features-bert-baseline.npy"), features.numpy())
    np.save(join(in_dir, "labels-bert-baseline.npy"), np.array(labels))
    with open(join(in_dir, "ids-bert-baseline.txt"), 'w') as f:
        f.write('\n'.join(ids) + '\n')
    return ids, features, labels

def split_input(ids, features, labels):
    n_train = int(len(labels) * 0.9 * 0.8)
    n_valid = n_train + int(len(labels) * 0.9 * 0.2)
    train = NewsDataset(ids[:n_train], features[:n_train], labels[:n_train])
    valid = NewsDataset(ids[n_train:n_valid], features[n_train:n_valid], labels[n_train:n_valid])
    test = NewsDataset(ids[n_valid:], features[n_valid:], labels[n_valid:])
    return train, valid, test

def read_politifact_input(in_dir):
    rumorities = {'real': REAL, 'fake': FAKE}
    inpt = []
    for rumority, label in rumorities.items():
        for news_id in tqdm(listdir(join(in_dir, rumority)), desc=f'politifact-{rumority}'):
            content_fn = join(in_dir, rumority, news_id, 'news content.json')
            if not isfile(content_fn): continue
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
            inpt.append([news_id, content['title'] + " [SEP] " + content["text"], label, other_features])
    return inpt

def read_politifact():
    in_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset/politifact'
    ids, features, labels = try_load_features(in_dir)
    if labels == None:
        inpt = read_politifact_input(in_dir)
        ids, features, labels = preprocess_and_save(inpt, in_dir, max_seq_len=490)
    return split_input(ids, features, labels)

def read_pheme_input(in_dir):
    rumorities = {'non-rumours': REAL, 'rumours': FAKE}
    inpt = []
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
                        inpt.append([tweet["id_str"], tweet['text'], label, other_features])
    return inpt

def read_pheme():
    in_dir = '/rwproject/kdd-db/20-rayw1/pheme-figshare'
    ids, features, labels = try_load_features(in_dir)
    if labels == None:
        inpt = read_pheme_input(in_dir)
        ids, features, labels = preprocess_and_save(inpt, in_dir, max_seq_len=49)
    return split_input(ids, features, labels)

def read_buzzfeed_input():
    rumorities = {'real': REAL, 'fake': FAKE}
    inpt = []
    for rumority, label in rumorities.items():
        df = pd.read_csv(join(in_dir, f"BuzzFeed_{rumority}_news_content.csv"))
        for _, row in df.iterrows():
            feature = [
                int(str(row.movies) == 'nan'), 
                int(len(row.images.split(','))) if str(row.images) != 'nan' else 0
            ]
            source = [0 for i in range(len(sources))]
            s = row['source'] if str(row['source']) != 'nan' else ''
            source[sources_map[s]] = 1
            feature.extend(source)
            inpt.append([row['id'], row['title'] + ' [SEP] ' + row['text'], label, feature])
    return inpt

def read_buzzfeed():
    sources = ['', 'http://author.groopspeak.com', 'http://conservativetribune.com', 'http://www.yesimright.com', 'http://clashdaily.com', 'https://goo.gl', 'http://occupydemocrats.com', 'http://www.thepoliticalinsider.com', 'http://politi.co', 'http://theblacksphere.net', 'http://allenwestrepublic.com', 'http://www.ifyouonlynews.com', 'http://eaglerising.com', 'http://www.proudcons.com', 'http://author.addictinginfo.org', 'http://www.opposingviews.com', 'http://abcn.ws', 'http://freedomdaily.com', 'http://addictinginfo.org', 'http://usherald.com', 'http://cnn.it', 'https://www.washingtonpost.com', 'http://conservativebyte.com', 'http://100percentfedup.com', 'http://winningdemocrats.com', 'http://rightwingnews.com', 'https://ihavethetruth.com', 'http://www.chicksontheright.com', 'http://www.addictinginfo.org']
    sources_map = {s: sid for sid, s in enumerate(sources)}
    in_dir = '/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle/'
    ids, features, labels = try_load_features(in_dir)
    if labels == None:
        inpt = read_buzzfeed_input()
        ids, features, labels = preprocess_and_save(inpt, in_dir, max_seq_len=490)
    return split_input(ids, features, labels)

