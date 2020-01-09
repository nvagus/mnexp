# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import utils
import re

class Impression:
        __slots__ = ['pos', 'neg']

        def __init__(self, d):
            d = d.split('#TAB#')
            self.pos = [int(k) for k in d[0].split(' ')]
            self.neg = [int(k) for k in d[1].split(' ')]

        def negative_samples(self, n):
            return np.random.choice(self.neg, n)
userpos = []
userneg = []
with open('../data/ClickData.tsv','r') as file:
    for line in file:
        line = line.strip('\n').split('\t')
        if line[2]:
            ih = [Impression(d) for d in line[2].split('#N#') if not d.startswith('#TAB#') and not d.endswith('#TAB#')]
            poslen = []
            neglen = []
            for im in ih:
                poslen.append(len(im.pos))
                neglen.append(len(im.neg))
            
            userpos.append(poslen)
            userneg.append(neglen)

userposlen = []
for pos in userpos:
    userposlen.append(np.sum(pos))

userneglen = []
for neg in userneg:
    userneglen.append(np.sum(neg))           

print('user clicked news history count,即每个用户历史点击新闻个数')
print(pd.Series(userposlen).describe())
print('90%:',pd.Series(userposlen).quantile(q=0.9))
print('95%:',pd.Series(userposlen).quantile(q=0.95))
print('99%:',pd.Series(userposlen).quantile(q=0.99))

print('user negative count,即每个用户负样本个数')        
print(pd.Series(userneglen).describe())
print('90%:',pd.Series(userneglen).quantile(q=0.9))
print('95%:',pd.Series(userneglen).quantile(q=0.95))

allimppos = []
for pos in userpos:
    allimppos.extend(pos)     


print('user impression positive count,即每个impression正样本个数')     
print(pd.Series(allimppos).describe())
print('90%:',pd.Series(allimppos).quantile(q=0.9))
print('95%:',pd.Series(allimppos).quantile(q=0.95))
print('99%:',pd.Series(allimppos).quantile(q=0.99))
allimpneg = []
for neg in userneg:
    allimpneg.extend(neg)     


print('user impression negative count,即每个impression负样本个数')     
print(pd.Series(allimpneg).describe())
print('90%:',pd.Series(allimpneg).quantile(q=0.9))
print('95%:',pd.Series(allimpneg).quantile(q=0.95))
print('99%:',pd.Series(allimpneg).quantile(q=0.99))
allimp = []
for user in userpos:
    allimp.append(len(user))   


print('user impression count,即每个用户impression个数')
print(pd.Series(allimp).describe())
print('90%:',pd.Series(allimp).quantile(q=0.9))
print('95%:',pd.Series(allimp).quantile(q=0.95))
print('99%:',pd.Series(allimp).quantile(q=0.99))

punc_idx = []
with utils.open('../data/PuncIndex.tsv','r') as file:
    for line in file:
        punc_idx.append(line.strip('\n').split('\t'))
puncs = [punc[1] for punc in punc_idx if punc[0] == '.' or punc[0] == '!' or punc[0] == '?']
puncs = set(puncs)


cnt_arr = []
with utils.open('../data/DocMeta.tsv','r') as file:
    for line in file:
        line = line.strip('\n').split('\t')
        title = line[4].split(' ')
        body = line[5].split(' ')
        title_len = len(title)
        
        body_len = len(body)
        
        sent_cnt = []
        lst = -1
        for i,token in enumerate(body):
            if token in puncs:
                sent_cnt.append(i-lst)
                lst = i
        
        if lst != i:
            sent_cnt.append(body_len-1-lst)
               
        cnt_arr.append([title_len,body_len,sent_cnt])


tb = pd.DataFrame([cnt[0:2] for cnt in cnt_arr])
tb.columns = ['title','body']
print(tb.describe())

body_sent_cnt = pd.DataFrame([len(cnt[2]) for cnt in cnt_arr])
body_sent_cnt.columns = ['sent_cnt']
print(body_sent_cnt.describe())

all_sent = []
for cnt in cnt_arr:
    all_sent.extend(cnt[2])

sent_len = pd.DataFrame(all_sent)
sent_len.columns = ['sent_len']
print(sent_len.describe().astype('int'))

