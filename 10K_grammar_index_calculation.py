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
    '''
    SEC filings include HTML & XBRL codes.
    There is no perfect solution to clean the mess up
    '''
    replacements = [
        ('&(.{2,6});', ' '), # Replace all Unicode strings
        ('<.*?>', ''), # Remove all HTML tags
        ('\n', ' '),
        ('\t', ' '),
        ('\r', ' ')
    ]
    for old, new in replacements:
        text = re.sub(old, new, text)
    text = re.sub(' +', ' ', text)
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




#== Test ground ====================================================================
# reference: https://github.com/EricHe98/Financial-Statements-Text-Analysis/blob/master/Documentation/Cleaning-Raw-Filings.md

df_us_10k_gram_index = pd.read_csv('D:/논문-Future Tense Mining/data/snp_10k.csv') ### *** --- *** YOU NEED TO DEFINE *** --- *** ###
sec_path = r'D:/논문-Future Tense Mining/data/SEC_10K/' ### *** --- *** YOU NEED TO DEFINE *** --- *** ###

adsh = df_us_10k_gram_index.adsh[0]
string = file_to_string(sec_path + adsh + '.txt')

import re

# Step 4: Return everything between the 10-K and the tags
start = re.search('(?<=<TYPE>10-K).*?', string).start()
end = re.search('.*?(?=</TEXT>)', string).end()
string = string[start:end]

# Step 5
string = re.sub('(?<=<TYPE>).*?(?=\n\<)', '', string)
string = re.sub('<TYPE>', '', string)

# Steop 6
string = re.sub('(?<=<SEQUENCE>).*?(?=\n\<)', '', string)
string = re.sub('<SEQUENCE>', '', string)
string = re.sub('(?<=<FILENAME>).*?(?=\n\<)', '', string)
string = re.sub('<FILENAME>', '', string)
string = re.sub('(?<=<DESCRIPTION>).*?(?=\n\<)', '', string)
string = re.sub('<DESCRIPTION>', '', string)

start2 = re.search('<head>', string).start()
end2 = re.search('</head>',string).end()
string = string[:start2] + string[end2:]

start3 = re.search('<(table)', string).start()
end3 = re.search('</table>',string).end()
string = string[:start3] + string[end3:]

# Step 7: Insert ">Â°Item" symbol
pattern = r'>Item( \d{1,2}\D)|>Item(\d{1,2}\D)|>ITEM( \d{1,2}\D)|>ITEM(\d{1,2}\D)'
temp = [(m.start(0), m.end(0)) for m in re.finditer(pattern, string)]
for s,e in temp:
    print(string[s-30:e+30])
    print('\n')

string = re.sub(pattern, r'>Â°Item\g<0>', string)

# Step 8: Remove all HTML tags
string = re.sub('<.*?>', '', string)

# Step 9: Replace all Unicode strings
string = re.sub('&(.{2,6});', ' ', string)

# Step 10: Replace multiple spaces with a single space
string = string.replace('\n',' ')
string = string.replace('\t',' ')
string = re.sub(' +', ' ', string)

# Check results
sections = string.split('Â°Item')

