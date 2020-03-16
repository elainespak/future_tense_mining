# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:47:21 2020

@author: elain
"""

import pandas as pd
import numpy as np
import os
import math
import json
import itertools
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import langid
from ast import literal_eval

stemmer = PorterStemmer()

# Define functions
def to_One_List(lists):
    '''
    # list of lists to one list , e.g. [[1,2],[3,4]] -> [1,2,3,4]
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    '''
    return list(itertools.chain.from_iterable(lists))

class Sentence_info:
    def __init__(self, sent, K):
        '''
        # INPUT
        # sent: num_sent (e.g., '-Amazing people' --> [0, 1]) ###['amaz', 'peopl'])
        
        # DEFINED
        # word_frequency: occurrence count of each word in sent 
        # unique_count: count of unique words in sent   < ---- 안쓰임!!!!! 대체 뭐
        # label: initially, set to -1
        '''
        self.word_frequency = FreqDist(sent)
        self.unique_word_count = len(self.word_frequency)
        self.aspect_label = np.array([-1]*K)
        
def sent_aspect_match(sent, aspects, K=1):
    '''
    # INPUT
    # aspects: list of list of aspects
               e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']]
    # k: number of different aspects
    
    # OUTPUT
    # match_count: k-dimensional vector representing the number of aspect words in the review
    '''
    match_count = np.zeros(K)
    sent_info = Sentence_info(sent, K)
    for idx in range(K):
        for word_num, word_num_count in sent_info.word_frequency.items():
            if word_num in aspects[idx]:
                match_count[idx] += word_num_count
    return match_count

def ChisqTest(N, taDF, tDF, aDF):
    '''
    # INPUT
    # N: all sentences that have some sort of aspect label
    # taDF: term in the aspect-labeled Document Frequency
    # tDF: term Document Frequency
    # aDF: aspect-labeled Document Frequency
    Calculate Chi-Square
    '''
    A = taDF  ## term & aspect
    # A+B = tDF
    B = tDF - A # term occurring in non-aspect Document Frequency
    C = aDF - A # number of sentences without the term
    D = N - A - B - C 
    return ((N * ( A * D - B * C )**2)) / ((aDF * ( B + D ) * tDF * ( C + D )) + 0.00001)

def collect_stat_for_each_review(report, aspects, Vocab):
    '''
    # INPUT
    # report: each report
    # aspects: list of list of aspects
               e.g., [['pay','money','benefits'], ['coworkers','team','colleagues']]
    '''
    K = len(aspects)
    report.num_stn_aspect_word = np.zeros((K,report.NumOfUniWord))
    report.num_stn_aspect = np.zeros(K)
    report.num_stn_word = np.zeros(report.NumOfUniWord)
    report.num_stn = 0
    
    for Sentence in report.Sentences_info:
        for idx in range(K):
            if Sentence.aspect_label[idx] == 1:   # if the sentence has an aspect label,
                report.num_stn += 1
                for l in Sentence.aspect_label:
                    report.num_stn_aspect[idx] += 1
                for w,v in Sentence.word_frequency.items():#keys():
                    z = np.where(w == report.UniWord)[0]  # index? 0?
                    report.num_stn_word[z] += v
                for l in Sentence.aspect_label:
                    for w,v in Sentence.word_frequency.items():#keys():
                        z = np.where(w == report.UniWord)[0] # index? 0?
                        report.num_stn_aspect_word[idx,z] += v
    return report.num_stn_aspect_word, report.num_stn_aspect, report.num_stn_word, report.num_stn

def label_sentence_UseVocab(only_sentences, VocabDict):
    '''
    Label every word of a sentence by using:
    1) a corresponding number from the VocabDict (vocabulary lookup table)
        OR
    2) "None" label
    
    '''
    num_sent = []
    for sent in only_sentences:
        temp = [VocabDict.get(w) for w in sent]
        #temp = [-1 if w == None else w for w in temp]
        if len(temp) > 0:
            num_sent.append(temp)
    return num_sent
           
class Report:
    def __init__(self, ind_data, VocabDict, K):
        '''
        # INPUT
        # ind_data: each individual review
        # VocabDict: vocabulary lookup table
        
        # DEFINED
        # Sentence_class: class Sentences
        '''
        self.Sentences_in_nums = label_sentence_UseVocab(ind_data, VocabDict)
        self.Sentences_info = [Sentence_info(sent_in_nums, K) for sent_in_nums in self.Sentences_in_nums]
        UniWord = {}
        for sent in self.Sentences_info:
            UniWord = UniWord | sent.word_frequency.keys()
        UniWord = {-1 if w==None else w for w in UniWord}
        self.UniWord = np.array([w for w in UniWord])
        self.UniWord.sort()
        self.NumOfUniWord = len(self.UniWord)
        
class Report_Collection:
    def __init__(self, dat, K):
        # get each report
        self.Reports = [Report(dat['Nouns'][idx], vocab_dict, K) for idx in range(len(dat))]

def save_Aspect_Keywords_to_file(filepath, Vocab):
    '''
    # INPUT
    # filepath: path where the complete aspect words text file will locate
    '''
    f = open(filepath, 'w')
    for w in aspect_terms[0]:
        try:
            f.write(Vocab[w]+", ")
        except:
            pass
    f.close()


# Setup
path = r'C:/Users/elain/Desktop/논문-Future Tense Mining/data/'
aspect_type = 'future2_'
input_name = 'aspect_seed_words.txt'
output_name = 'complete_aspect_words.txt'
dat = pd.read_csv(path + 'kospi200_2011_2015_nouns.csv', encoding='utf-8')
dat['Nouns'] = dat['Nouns'].apply(lambda x: literal_eval(x))

######## temporary fix ########
drop_nums = []
for i in range(len(dat)):
    if len(dat['Nouns'][i]) == 0:
        drop_nums.append(i)
dat = dat.drop(dat.index[drop_nums])
dat = dat.reset_index(drop=True)
########

# Make master vocabulary
words = []
for n in dat['Nouns']:
    n_unlisted = to_One_List(n)
    words += n_unlisted
freq = FreqDist(words) # 21,115 unique nouns
vocab = [k for k,v in freq.items() if v > 0]
vocab_dict = dict(zip(vocab, range(len(vocab))))

# Load seed words
aspect_terms = []

f = open(path + aspect_type + input_name, 'r', encoding='utf-8')
for line in f:
    aspect_terms.append([vocab_dict.get(stemmer.stem(w.strip().lower())) for w in line.split(",")])
f.close()
print("-------- Aspect Keywords loading completed!")

# Add sentence labels with seed words
NumIter = 5
max_num = 5
K = len(aspect_terms)
V = len(vocab)
aspect_num = 0

all_reports = Report_Collection(dat, K).Reports

for report in all_reports:
    for i in range(len(report.Sentences_in_nums)):
        match_count = sent_aspect_match(report.Sentences_in_nums[i], aspect_terms, 2)
        for idx in range(K):        
            if np.max(match_count[idx])>0: # if at least one of the aspects has a match
                report.Sentences_info[i].aspect_label[idx] = np.ones(1)
            else:
                report.Sentences_info[i].aspect_label[idx] = np.zeros(1)

# Run iterations

# Load seed words
aspect_terms = []

f = open(path + aspect_type + input_name, 'r', encoding='utf-8')
for line in f:
    aspect_terms.append([vocab_dict.get(stemmer.stem(w.strip().lower())) for w in line.split(",")])
f.close()
print("-------- Aspect Keywords loading completed!")

for i in range(NumIter):
    all_num_stn_aspect_word = np.zeros((K,V))
    all_num_stn_aspect = np.zeros(K)
    all_num_stn_word = np.zeros(V)
    all_num_stn = 0
    for report in all_reports:
        report.num_stn_aspect_word, report.num_stn_aspect, report.num_stn_word, report.num_stn = collect_stat_for_each_review(report, aspect_terms, vocab)
        all_num_stn += report.num_stn # total number of sentences with any aspect label
        all_num_stn_aspect += report.num_stn_aspect

        for w in report.UniWord:
            z = np.where(w == report.UniWord)[0][0] # index, since the matrix for review is small
            all_num_stn_word[w] += report.num_stn_word[z] # number of times aspect_i words (z) appear in all sentences
            all_num_stn_aspect_word[:,w] += report.num_stn_aspect_word[:,z]

    Chi_sq = np.zeros((K,V))
    for k in range(K):
        for w in range(V):
            Chi_sq[k,w] = ChisqTest(
                        all_num_stn, # sentences with any aspect
                        all_num_stn_aspect_word[k,w], # num. of words in sentences belonging to aspect_k
                        all_num_stn_word[w], # num. of word occurrence in any sentences
                        all_num_stn_aspect[k] # num. of sentences of aspect_k
                        )

    for idx in range(Chi_sq.shape[0]):
        cs = Chi_sq[idx]
        x = cs[np.argsort(cs)[::-1]] # descending order
        y = np.array([not math.isnan(v) for v in x]) # return T of F
        words = np.argsort(cs)[::-1][y] #
        
        for w in words:
            if w not in to_One_List(aspect_terms):
                aspect_terms[idx].append(w)
                aspect_num += 1
            if aspect_num > max_num:
                break
        
        aspect_num = 0
    print("complete iteration "+str(i+1)+"/"+str(NumIter))

print(aspect_terms)
vocab_dict2 = {v:k for k,v in vocab_dict.items()}
print([vocab_dict2.get(w) for w in aspect_terms[0]])
print([vocab_dict2.get(w) for w in aspect_terms[1]])

# Save
save_Aspect_Keywords_to_file(path + aspect_type + output_name, vocab)

