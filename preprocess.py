# -*- coding: utf-8 -*-

import utils
import pandas as pd
import pickle

df = pd.read_csv('../data/Vocab.tsv',sep='\t',names=['word','index','vec'])

punc_idx = [
('.',df[df['word']=='.']['index'].values[0]),
('!',df[df['word']=='!']['index'].values[0]),
('?',df[df['word']=='?']['index'].values[0]),
(';',df[df['word']==';']['index'].values[0]),
(',',df[df['word']==',']['index'].values[0])
]

with utils.open('../data/PuncIndex.tsv','w') as file:
    for punc,idx in punc_idx:
        file.write(punc+'\t'+str(idx)+'\n')

vocab2idx = {word:index for word,index,_ in df.values}

pickle.dump(vocab2idx,open('../data/vocab2idx.pkl','wb'))