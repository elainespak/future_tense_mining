# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:28:50 2020

@author: elain
"""

# Setup
from os import listdir
from os.path import join
from bs4 import BeautifulSoup
from nltk import sent_tokenize
import pickle

folder_path = r'C:/Users/elain/Desktop/논문-Future Tense Mining/data/reuters21578/'

def process_news(filename):
    '''
    Bring the body content from each .sgm file
    '''
    document = BeautifulSoup(open(filename), 'html.parser', from_encoding="utf-8")
    body = document.find_all('body')
    return body

def text_preprocess(body):
    '''
    Clean raw text
    '''
    raw_text = [b.get_text()[:-10] for b in body]
    text = [rt.replace('\n', ' ') for rt in raw_text]
    sents = []
    for t in text:
        s = sent_tokenize(t)
        sents += s
    return sents

# Save every sentence from the Reuters file into a single list
all_sents = []
for f in listdir(folder_path):
    if f.endswith('.sgm'):
        try:
            file = join(folder_path, f)
            text = process_news(file)
            sents = text_preprocess(text)
            all_sents += sents
        except:
            print(f'error occured for file {f}')

# Check
print(len(all_sents))        
print(all_sents[1002])

# Save
with open(join(folder_path,'reuters_sentences.pickle'), 'wb') as f:
    pickle.dump(all_sents, f)
