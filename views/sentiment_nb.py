from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pandas as pd

stoplist = stopwords.words('english')

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(sentence))]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)

    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' reviews')
    print ('Test set size = ' + str(len(test_set)) + ' reviews')

    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

df = pd.read_csv('../data/current_data.csv')
data = []

for index,rows in df.iterrows():
    a = (rows['Review'],rows['Useful'])
    data.append(a)

corpus_features = [(get_features(each,''),label) for (each,label) in data]
print ('Collected ' + str(len(corpus_features)) + ' feature sets')

train_set, test_set, classifier = train(corpus_features, 1.0)

def predict_text(text):
    inp1 = "Ajay is good professor. I listen to his lectures frequently"
    inp_featureset = get_features(inp1,'')

    return classifier.classify(inp_featureset)
