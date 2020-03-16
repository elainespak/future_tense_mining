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
import pickle

local_path = 'C:/Users/elain/Desktop/논문-Future Tense Mining/'
data_folder = 'data/'
news_list = [pd.read_excel(local_path + data_folder + '3_문어체_뉴스(1)_200226.xlsx'),
             pd.read_excel(local_path + data_folder + '3_문어체_뉴스(2)_200226.xlsx'),
             pd.read_excel(local_path + data_folder + '3_문어체_뉴스(3)_200226.xlsx'),
             pd.read_excel(local_path + data_folder + '3_문어체_뉴스(4)_200226.xlsx')
             ]
dat = pd.concat(news_list)
dat = dat.reset_index(drop=True)
sent_text = dat['원문']


############## ---------------------- ###############
### --- *** --- Extracting Nouns --- *** --- ###
############## ---------------------- ###############
    
# Create object instance
okt = Okt()
print(sent_text[0])
print(okt.nouns(sent_text[0])) # works!

noun_text = sent_text.apply(lambda x: okt.nouns(x))
print(noun_text[:20])

# Save file
with open(local_path + data_folder + '문어체_뉴스_nouns.pickle', 'wb') as handle:
    pickle.dump(noun_text, handle, protocol=pickle.HIGHEST_PROTOCOL)
