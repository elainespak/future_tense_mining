# -*- coding: utf-8 -*-


#== All packages ======================================================================

import re
from nltk.tokenize import sent_tokenize
import pandas as pd



#== Pre-process to get sentences ======================================================

#-- Define necessary functions --------------------------------------------------------
def file_to_string(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
          data = f.read()
    return data

def pre_process_raw_text(text):
    replacements = [
        ('&#8217', '\''),
        (' \[Text Block\]', '.'),
        ('No. ', 'No.'),
        ('<.*?>', ''),
        ('\s\s+', ' '),
        (';|\n|\t|\r|&#160|&#8221|&#8220', ' '),
        # added
        ('&amp|&gt|&lt', ' '),
        ('FONT-SIZE: \d+pt', ' ')
    ]
    for old, new in replacements:
        text = re.sub(old, new, text)
    return text

def get_clean_sentence(string):
    real_sents = [sent for sent in sent_tokenize(string) if len(sent.split())>5]
    clean_sents = [sent for sent in real_sents if sum(map(str.islower, sent)) > len(sent)/3]
    return clean_sents

def get_grammar_index(sentences, grammatic_tokens):
    '''
    count sentences with at least one grammatic_tokens occurence
    '''
    c = 0
    c_false = 0
    temp = 0
    total = len(sentences)
    
    for sent in sentences:
        for g in grammatic_tokens:
            if g in sent:
                temp += 1
                c_false += 1
        if temp > 0:
            c += 1
            temp = 0
    return total, c, c/total*100, c_false, c_false/total*100


#== Compile & Run ====================================================================

df_us_10k_gram_index = pd.read_csv('D:/논문-Future Tense Mining/data/snp_10k.csv') ### *** --- *** YOU NEED TO DEFINE *** --- *** ###
sec_path = r'D:/논문-Future Tense Mining/data/SEC_10K/' ### *** --- *** YOU NEED TO DEFINE *** --- *** ###

grammatic_tokens = ['will', 'shall', 'potential', 'prospective', #'plan',
                    'proposal', 'expect', 'likely', 'believe', 'feel', 'forecast', 'projection', 'intend']

error_open = []
error_pp = []
error_clean = []

grammar_dict = {}


for idx, row in df_us_10k_gram_index.iterrows():
    try:
        string = file_to_string(sec_path + row.adsh + '.txt')
    except:
        print(f'error opening file {row.adsh}')
        error_open.append(row.adsh)
    try:
        string = pre_process_raw_text(string)
    except:
        print(f'error pre-processing file {row.adsh}')
        error_pp.append(row.adsh)
    try:
        sents = get_clean_sentence(string)
        grammar_dict[row.adsh] = get_grammar_index(sents, grammatic_tokens)
    except:
        print(f'error pre-processing file {row.adsh}')
        error_clean.append(row.adsh)
    if idx % 100 == 0:
        print(idx)
        
        
df = pd.DataFrame.from_dict(grammar_dict, orient='index')
df.columns = ['total', 'future_grammar_sents', 'future_grammar_perc', 'false_sents', 'false_perc']

#-- Save ----------------------------------------------------------------------------
output_name = '1' ### *** --- *** YOU NEED TO DEFINE *** --- *** ###
df.to_csv(output_name+'.csv')

#-- Check errors --------------------------------------------------------------------
print(f'Error opening these files: {error_open}')
print(f'Error pre-processing these files: {error_pp}')
print(f'Error cleaning these files into sentences: {error_clean}')


