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
import json
import sklearn
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import sports_match

from tensorflow.keras.models import load_model
from attention import AttentionWithContext



def load_local_dict(path):
    ''' 
    input: file_name './dict/label.dict' one label one line
    output: diction k is label while v is row num -1
    '''
    res = {}
    cur = 0 
    for lines in open(path):
        data = lines.strip()
        res[data]=cur
        cur+=1
    return res



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
    
    doc_padded = get_padded_vec(doc = doc,token = token,lenth = lenth)
    return doc_padded


def rule_based(title,content,cur_res,label_dict):
    #covid->health
    res={}
    conv_list = ['covid','corona'] 
    if len(cur_res) == 0 :
        for k in conv_list:
            if k in title.lower():
                res['Health_PublicHealth'] = 1.0
                break
    else:
        if 'Sports' in cur_res:
            sports = sports_match.search(content)
            cur_sports = []
            for k in sports:
                cur_sports.append((k,len(sports[k])))
            cur_sports = sorted(cur_sports,key=lambda x:x[1],reverse=True)
            if len(cur_sports)>0 and  cur_sports[0][1]>3 and cur_sports[0][0] not in cur_res:
                cur_res[cur_sports[0][0]] = 1.0

        return cur_res


    return res

def regular_result(cur_res):
    res = {}
    first_cat = {}
    second_cat = {}
    third_cat = {}
    other = {}
    for k in cur_res:
        lenth = len(k.split('_'))
        if lenth == 1:
            first_cat[k] = cur_res[k]
        elif lenth == 2:
            second_cat[k] = cur_res[k]
        elif lenth == 3:
            third_cat[k] = cur_res[k]
    for k in third_cat:
        sec = '_'.join(k.split('_')[:2])
        if sec not in second_cat:
            second_cat[sec] = third_cat[k]
    for k in second_cat:
        first = k.split('_')[0]
        if first not in first_cat:
            first_cat[first] = second_cat[k]
    #add other
    for key in second_cat:
        if key not in str(third_cat):
            third_cat[key+'_Other'] = second_cat[key]
    for key in first_cat:
        if key not in str(second_cat):
            second_cat[key+'_Other'] = first_cat[key]
    
    res['first_cat'] = first_cat
    res['second_cat'] = second_cat
    res['third_cat'] = third_cat
    return res



def process(model,id,url,title,content,title_token,content_token,label_dict):
    '''
    main processing function：
    title_token :tokenizer
    '''
    #preprocess for padded vec
    content_padded = preprocess(content,content_token,200)
    title_padded = preprocess(title,title_token,30)
    
    #model result 
    py = model.predict([title_padded,content_padded])
    class_index = dict(zip(label_dict.values(),label_dict.keys()))
    for j in range(len(py)):
        res = {}
        for i in range(len(py[j])):
            if py[j][i]>0.5:
                res[class_index[i]] = float(py[j][i])
        
        #rule based
        res = rule_based(title[j],content[j],res,label_dict)

        #regular result
        res = regular_result(res)
        print(json.dumps({'text_category_v2':res,'_id':id[j],'url':url[j]}))


if __name__=='__main__':
    model = load_model('/mnt/nlp/text-classify-v2/model/lstm-att-complex-v4-6-11',
            custom_objects={'AttentionWithContext':AttentionWithContext})
    title_token = pickle.load(open('./dict/title_token.pcl','rb'))
    content_token = pickle.load(open('./dict/content_token.pcl','rb'))
    label_dict = load_local_dict('./dict/new.dict')
    path = sys.argv[1]
    content = []
    title = []
    id = []
    url = []
    for lines in open(path):
        try:
            data = json.loads(lines.strip())
            content_ = data['content'].lower()
            title_ = data['title'].lower()
            id_ = data['docid']
            url_ = data['url']
        except :
            continue
        content.append(content_)
        title.append(title_)
        id.append(id_)
        url.append(url_)
    process(model,id,url,title,content,title_token,content_token,label_dict)
