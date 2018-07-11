# -*- coding: utf-8 -*-
import re
import sys
import json
import pickle

URLS = []


def remove_punctuation(tweet):
    text = tweet.replace(".", " ")
    text = re.sub(r'(\d+):(\d+)', r"\1\2", text)
    text = re.sub(r'$(\d+)\.(\d+)', r"\1\2", text)
    text = tweet.replace("'s ", "s ")
    text = tweet.replace("i've ", "ive ")
    text = tweet.replace("i'd ", "id ")
    text = tweet.replace("i'm ", "im ")
    text = tweet.replace("i'll ", "ill ")
    text = text.replace(u"-", " ")
    text = text.replace("?", " ")
    text = text.replace("!", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace(";", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("'", " ")
    text = text.replace('"', " ")
    text = text.replace(u'–', " ")
    text = text.replace(u'--', ' ')
    text = text.replace(u'"', ' ')
    text = text.replace(u"...", " ")
    text = text.replace(u"’", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_case(tweet):
    return tweet.lower()

def clean_tweet(tweet):
    TWEET_OUT = []
    split_tweet = tweet.split()
    for n, i in enumerate(split_tweet):
        if 'http' in i:
            if i[-1] != u'…':
                try:
                    matches = re.findall(r"(.*?)?(\S+\:\/\/\S+\.\S+\/\S+)", i)
                    if matches[0][0] != u'':
                        t = normalize_case(tweet=t)
                        t = remove_punctuation(tweet=matches[0][0])
                        TWEET_OUT.append(t)
                    TWEET_OUT.append("^^URL^^")
                    URLS.append(matches[0][1])
                except:
                    TWEET_OUT.append("^^URL^^")
            else:
                TWEET_OUT.append("^^URL^^")
        else:
            t = remove_punctuation(tweet=i)
            t = normalize_case(tweet=t)
            TWEET_OUT.append(t)
    
    return ' '.join(TWEET_OUT)

def read_jsons(jsons):
        dataDX = {}
        for i in jsons:
            F = open(i, 'r').read()
            data = json.loads(F)
            for i in data:
                dataDX[i['id_str']] = i
                dataDX[i['id_str']]['text'] = clean_tweet(tweet=dataDX[i['id_str']]['text'])
        return dataDX

def write_urls(path):
    OUT = open(path, 'wb')
    pickle.dump(URLS, OUT)
    OUT.close()

if __name__ == "__main__":
    FAILED = 0
    SOURCE = ["Data/trump_tweets_06272018.json"]
    read_jsons(jsons=SOURCE)
    write_urls(path="/Users/khambert/hgWD/unreal45/Data/urls.pkl")
    