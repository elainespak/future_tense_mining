# -*- coding: utf-8 -*-


import nltk
import pickle
#nltk.download('propbank')
#nltk.download('treebank')
from tqdm import tqdm
from nltk.corpus import treebank
from nltk.corpus import propbank


# Tiny example: PropBank
pb_instances = propbank.instances()
len(pb_instances) # 112917
inst = pb_instances[1]
inst.fileid, inst.sentnum, inst.wordnum
print(propbank.instances()[1])
infl = inst.inflection
infl.form, infl.tense, infl.aspect, infl.person, infl.voice

# Tiny example: TreeBank
len(treebank.fileids()) # 199
len(treebank.parsed_sents()) # 3914
print(treebank.words('wsj_0001.mrg')[:])

# Compile all propbank metadata of verbs
pb_instances = propbank.instances()
index = [(inst.fileid, inst.sentnum, inst.wordnum, inst.inflection.tense) for inst in tqdm(pb_instances)]

ann = []
for fileid, sentnum, wordnum, tense in tqdm(index):
    allwords = treebank.parsed_sents(fileid)[sentnum].leaves()
    word = allwords[wordnum]
    ann.append((fileid, sentnum, wordnum, tense, word, allwords))
    
with open('propbank_preprocessed.pkl', 'wb') as f:
    pickle.dump(ann, f)