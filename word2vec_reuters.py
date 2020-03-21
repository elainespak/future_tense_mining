# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:36:56 2020

@author: elain
"""

# Setup
import pickle
import re
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import os
os.chdir(r'C:/Users/elain/Documents/GitHub/future_tense_mining/data/')

############# ------------ *** ------------ *** -------------- ##############
#############            Build my own word2vec model           ##############
############# ------------ *** ------------ *** -------------- ##############

# Define functions
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

def build_phrases(sentences):
    phrases = Phrases(sentences,
                      min_count=5,
                      threshold=7,
                      progress_per=1000)
    return Phraser(phrases)

def sentence_to_n_grams(phrases_model, sentence):
    return phrases_model[sentence]

def sentences_to_bi_grams(n_grams, input_file_name, output_file_name):
    with open(input_file_name, 'rb') as f:
        text = pickle.load(f)
        with open(output_file_name, 'w+') as out_file:
            for sentence in text:
                cleaned_sentence = clean_sentence(sentence)
                tokenized_sentence = tokenize(cleaned_sentence)
                parsed_sentence = sentence_to_n_grams(n_grams, tokenized_sentence)
                out_file.write(parsed_sentence + '\n')           

# Parameters
file_name = 'reuters_sentences.pickle'
output_name = 'reuters_bigrams.pickle'

# Open file
with open(file_name, 'rb') as f: # Unpickling
    text = pickle.load(f)

# Text pre-processing
cleaned_text = [clean_sentence(t) for t in text]
tokenized_text = [t.split() for t in cleaned_text]

# Build n-gram models
phrases_model = build_phrases(tokenized_text)
phrases_model.save('phrases_model_bigrams.txt')
bigram_text = [sentence_to_n_grams(phrases_model, t) for t in tokenized_text]

phrases_model2 = build_phrases(bigram_text)
phrases_model2.save('phrases_model_trigrams.txt')
trigram_text = [sentence_to_n_grams(phrases_model2, t) for t in bigram_text]

# Run Word2Vec with the corpus
model = Word2Vec(trigram_text, min_count=1, size= 50, workers=3, window=3, sg=1)

# Check results
print(model.most_similar('expect'))
print('\n')
print(model.most_similar('expected'))
print('\n')
print(model.most_similar('expecting'))
print('\n')
print(model.most_similar('future'))



############# ------------ *** ------------ *** -------------- ##############
#############             Use pretrained NLP models            ##############
############# ------------ *** ------------ *** -------------- ##############

# NOTE: STILL BUILDING & TESTING!


import gensim.downloader as api
from os import isfile, join

info = api.info()  # show info about available models/datasets

### Word2Vec
# If the model has never been loaded:
model = api.load('word2vec-google-news-300')
model.save('word2vec-google-news-300.model')

# If the model is already saved:
model = model.load('word2vec-google-news-300.model.model')
model.most_similar('expecting')


### Glove
# If the model has never been loaded:
model = api.load('glove-twitter-100')
model.save('glove-twitter-100.model')

# If the model is already saved:
model = model.load('glove-twitter-100.model')
model.most_similar('expecting')


### FastText
# If the model has never been loaded:
model = api.load('fasttext-wiki-news-subwords-300')
model.save('fasttext-wiki-news-subwords-300.model')

# If the model is already saved:
model = model.load('fasttext-wiki-news-subwords-300.model')
model.most_similar('expecting')


'''
def pretrained_model(name):
    file_name = join(name, '.model')
    if isfile(file_name):
        model = model.load(file_name)
    else:
        model = api.load(name)
    return model
'''
