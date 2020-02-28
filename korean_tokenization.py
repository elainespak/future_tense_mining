# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:30:20 2020

@author: elain
"""

############## ---------------------- ###############
### --- *** --- Setup --- *** --- ###
############## ---------------------- ###############

import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from konlpy.tag import Okt

local_path = 'C:/Users/elain/Desktop/Future Tense Mining/'
data_folder = 'data/'
output_folder = 'output_data/'
file_name = 'kospi200_2015_2019'
dat = pd.read_excel(local_path + data_folder + file_name + '.xlsx')
raw_text = dat['Text']


############## ---------------------- ###############
### --- *** --- Text Pre-processing --- *** --- ###
############## ---------------------- ###############

# Things to remove from the text
messy_items = ['\n', ',', '(', ')', '%', '~']

def clean_text(text):
    for i in messy_items:
        text = text.replace(i, '')    
    text = re.sub('\d+', '', text)
    return text

# Use nltk to split sentences (works great on Korean as well)
token_text = raw_text.apply(lambda x: sent_tokenize(x))

# Select relevant, proper sentences and pre-process
sent_text = []

for t in token_text:
    relevant_sentences = []
    
    for sent in t:
        if sent.endswith('니다.'):
            sent = clean_text(sent)
            relevant_sentences.append(sent)
            
    sent_text.append(relevant_sentences)

# Save checkpoint
dat['Relevant_Sentences'] = sent_text
dat.to_csv(local_path + output_folder + file_name + '_parsed.csv', index=False)
    

############## ---------------------- ###############
### --- *** --- Extracting Nouns --- *** --- ###
############## ---------------------- ###############
    
# Create object instance
okt = Okt()
print(sent_text[0][0])
print(okt.nouns(sent_text[0][0])) # works!

# Select relevant sentences and pre-process
def konlpy_noun(sentences):
    return [okt.nouns(sent) for sent in sentences]

sent_text = pd.Series(sent_text)
noun_text = sent_text.apply(lambda x: konlpy_noun(x))

# Save checkpoint
dat['Nouns'] = noun_text
dat.to_csv(local_path + output_folder + file_name + '_nouns.csv', index=False)





############## ---------------------- ###############
### --- *** --- Trash codes --- *** --- ###
############## ---------------------- ###############

text = raw_text.apply(lambda x: clean_text(x))


text = raw_text.apply(lambda x: x.replace('\n', ''))
text = text.apply(lambda x: x[10:]) # remove 'II. 사업의 내용 '
text = text.apply(lambda x: re.sub('[\d]{+}', '', x))

# Test nltk on Korean to split sentences
sent_tokenize(text[0]) # pretty good!


i = 0
for t in list(text):
    if 'II. 사업의 내용 ' in t:
        i+=1
print(i)


from soynlp.noun import LRNounExtractor
noun_extractor = LRNounExtractor()
nouns = noun_extractor.train_extract(sent_text[0][0])


from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

noun_extractor = LRNounExtractor_v2(verbose=True)
noun_extractor.train_extract(sent_text[0][:5])



from soynlp.word import WordExtractor

word_extractor = WordExtractor(min_frequency=7,
    min_cohesion_forward=0.05, 
    min_right_branching_entropy=0.0
)
word_extractor.train(sent_text[0]) # list of str or like
words = word_extractor.extract()

print(words)