import re
import sys
import csv
import time
import glob
import json
import keras
import codecs
import string
import gensim
import pprint
import os.path
import argparse
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM


# Command line arguments...
parser = argparse.ArgumentParser()
parser.add_argument('-test_app', '--test_app', action='store_true', help='test the app interface')
parser.add_argument('-source', '--source', type=str, help="Data source: [json|csv]")
parser.add_argument('-model_json_out', '--model_json_out', type=str, default='u45_model.json', help="Path to save model json")
parser.add_argument('-out_weights', '--out_weights', type=str, default='u45_weights.h5', help="Path to save model weights in h5")


# REFERENCES!
# http://www.trumptwitterarchive.com/archive
# https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b

# CONSTANTS!
MAX_LENGTH = 350

class U45(object):
    def on_epoch_end(self, epoch, _):
        # Save model...
        model_json = u45.model.to_json()
        with open(self.model_json_out, 'w') as json_file:
            json_file.write(model_json)
        sys.stderr.write("Saved model to {PATH}...\n".format(PATH=self.model_json_out))

        # Save weights...
        u45.model.save_weights(self.out_weights)
        sys.stderr.write("Saved weights to {PATH}...\n".format(PATH=self.out_weights))

        print '\nGenerating text after epoch: %d' % epoch
        texts = [
            'crooked',
            'crooked Hillary',
            'whale',
            'baseball',
            'friend',
        ]
        for text in texts:
            sample = self.generate_next(text=text)
            print "{TEXT} --> {SAMPLE}".format(TEXT=text, SAMPLE=sample)
        
    def generate_next(self, text, num_generated=10):
        word_idxs = []
        for word in text.lower().split():
            try:
                word_idxs.extend(self.word2idx(word))
            except:
                sys.stderr.write("{WORD} not found in corpus...\n".format(WORD=word))

        for i in range(num_generated):
            prediction = self.model.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in word_idxs)
    
    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def build_lstm(self):
        sys.stderr.write("Building lstm...")
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights]))
        self.model.add(LSTM(units=self.emdedding_size))
        self.model.add(Dense(units=self.vocab_size))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        sys.stderr.write("done.\n")

    
    def pre_lstm(self, sentences):
        X = np.zeros([len(sentences), MAX_LENGTH], dtype=np.int32)
        y = np.zeros([len(sentences)], dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence[:-1]):
                X[i, t] = self.word2idx(word)
            y[i] = self.word2idx(sentence[-1])
        return X, y

    def word2idx(self, word):
        return self.w2v_model.wv.vocab[word].index

    def idx2word(self, idx):
        return self.w2v_model.wv.index2word[idx]

    def train_w2v(self, sentences):
        sys.stderr.write("Training w2v...")
        self.w2v_model = gensim.models.Word2Vec(sentences, size=MAX_LENGTH, min_count=1, window=5, iter=100)
        self.pretrained_weights = self.w2v_model.wv.syn0
        self.vocab_size, self.emdedding_size = self.pretrained_weights.shape
        sys.stderr.write("done.\n")

    def clean_text(self, text):
        text = text.replace('&amp;', '&')
        #text = re.sub(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'URL', text, re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        # TODO: Capture urls and prevent them from getting normalized...
        #if urls != []:
        #    print urls
        
        #text = text.lower()
        #PUNCTUATION = '.,"\'?!-;:'
        #for i in PUNCTUATION:
        #    text = text.replace(i, '')
        text = text.encode('utf-8')
        return text.strip()

    def get_all_sentences(self):
        return [self.dataDX[i]['text'] for i in self.dataDX.keys()]

    def read_csvs(self, csvs):
        pass

    def read_jsons(self, jsons):
        dataDX = {}
        for i in jsons:
            F = open(i, 'r').read()
            data = json.loads(F)
            for i in data:
                dataDX[i['id_str']] = i
                dataDX[i['id_str']]['text'] = self.clean_text(text=dataDX[i['id_str']]['text']).split()
        return dataDX


    def __init__(self, data):
        super(U45, self).__init__()
        
        if os.path.basename(data[0]).split('.')[1] == 'json':
            self.dataDX = self.read_jsons(jsons=data)
        elif os.path.basename(data[0]).split('.')[1] == 'csv':
            self.dataDX = self.read_csvs(csvs=data)
        
        #pprint.pprint(self.dataDX)
        # print max([len(self.dataDX[i]['text']) for i in self.dataDX.keys()])
        # print [self.dataDX[i]['text'] for i in self.dataDX.keys() if len(self.dataDX[i]['text']) == 311]




if __name__ == "__main__":
    # Parse command line arguments...
    args = parser.parse_args()

    # Get data path...
    if args.source == 'json':
        source = '*.json'
    elif args.source == 'csv':
        source = '*.csv'
    DATA = glob.glob(os.path.join("Data", source))
    if args.test_app:
        u45 = U45(data=DATA)
        u45.model_json_out = args.model_json_out
        u45.out_weights = args.out_weights
        
        # Train w2v!
        # TODO: GET A SECOND TWITTER CORPUS TO INCLUDE IN THE W2V TRAINING
        u45.train_w2v(sentences=u45.get_all_sentences())
        
        # Train LSTM!
        X, y = u45.pre_lstm(sentences=u45.get_all_sentences())
        u45.build_lstm()
        sys.stderr.write("Training lstm...\n")
        u45.model.fit(X, y, batch_size=16, epochs=10, callbacks=[LambdaCallback(on_epoch_end=u45.on_epoch_end)])
        
        # Save model...
        model_json = u45.model.to_json()
        with open(args.model_json_out, 'w') as json_file:
            json_file.write(model_json)
        sys.stderr.write("Saved model to {PATH}...\n".format(PATH=args.model_json))

        # Save weights...
        u45.model.save_weights(args.out_weights)
        sys.stderr.write("Saved weights to {PATH}...\n".format(PATH=args.out_weights))