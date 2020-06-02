#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: process.py
# Author: NLP_zhaozhenyu
# Mail: zhaozhenyu_tx@126.com
# Created Time: 15:41:54 2019-01-25
#########################################################################
import sys
import re
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


def get_padded_vec(doc,token,lenth):
    #{'<sos>':1,'<eos>':2}
    def texts_to_sequences(input_x,vocab):
	res = []
	for x in input_x:
	    words = x.split(' ')
	    cur  = [1]
	    for word in words:
		if word in vocab:
		    cur.append(vocab[word]+3)
		else:
		    cur.append(0)
	    cur.append(2)
	    res.append(cur)
	return res
    
    vocab = token.vocabulary_
    
    word_ids = texts_to_sequences(doc,vocab)
    padded_input = pad_sequences(word_ids,maxlen=lenth,padding='post',truncating='post')
    
    return padded_input


def preprocess(doc,token,lenth):
    '''
    preprocessing content
    length is pad doc
    return:
        padded doc
    '''
    doc = doc.lower()
    doc_padded = get_padded_vec(doc = [doc],token = token,lenth = lenth)
    return doc_padded

def process(model,title,content,title_token,content_token,label_dict):
    '''
    main processing function：
    title_token :tokenizer
    '''
    #preprocess for padded vec

    res = {}
    content_padded = preprocess(content,content_token,200)
    title_padded = preprocess(title,title_token,30)

    py = model.predict([title_padded,content_padded])[0]

    class_index = dict(zip(label_dict.values(),label_dict.keys()))

    for i in range(len(py)):
        if py[i]>0.5:
            res[class_index[i]] = py[i]

    return {'category_classification_v2':res}
