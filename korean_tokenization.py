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

local_path = 'C:/Users/elain/Desktop/논문-Future Tense Mining/'
data_folder = 'data/'
output_folder = 'output_data/'
file_name = 'kospi200_2011_2015'
dat = pd.read_excel(local_path + data_folder + file_name + '.xlsx')
raw_text = dat['Text']


############## ---------------------- ###############
### --- *** --- Text Pre-processing --- *** --- ###
############## ---------------------- ###############

# Use nltk to split sentences (works great on Korean as well)
token_text = raw_text.apply(lambda x: sent_tokenize(x))

# Use extra hand-made parser
def additional_sent_tokenize(text):
    '''
    Some bullet points, which should be taken as separate sentences,
    are not accounted for by the nltk sentence tokenizer.
    This additional tokenizer will take care of such bullet points.
    '''
    if '※' in text:
        text_list = text.split('※')
    elif '* ' in text:
        text_list = text.split('* ')
    else:
        text_list = [text]
    return text_list

additional_token_text =[]

for t in token_text:
    temp_list = []
    for sent in t:
        temp_list.append(additional_sent_tokenize(sent))
    additional_token_text.append([item for sublist in temp_list for item in sublist])

# Select relevant, proper sentences and pre-process
def clean_text(text):
    '''
    Clean the messy raw text
    1) Remove \n, 전자공시시스템 dart.fss.or.kr, Page #
    2) Remove white space before and after the input text
    '''
    text = re.sub("\\n|(전자공시시스템 dart\.fss\.or\.kr)?(\n)?(Page \d{1,2})?(?=\s)",
                  " ",
                  text)
    text = text.strip()
    return text

sent_text = []

for t in additional_token_text:
    relevant_sentences = []
    
    for s in t:
        if s.endswith('니다.'):
            s = clean_text(s)
            s = s.strip()
            relevant_sentences.append(s)
            
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
print(okt.nouns(sent_text[0][0])) # Not bad

# Select relevant sentences and pre-process
def konlpy_noun(sentences):
    '''
    Return a list of nouns from a list of sentences
    '''
    return [okt.nouns(sent) for sent in sentences]

sent_text = pd.Series(sent_text)
noun_text = sent_text.apply(lambda x: konlpy_noun(x))

# Save checkpoint
dat['Nouns'] = noun_text
dat.to_csv(local_path + output_folder + file_name + '_nouns.csv', index=False)

