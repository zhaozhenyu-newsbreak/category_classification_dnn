#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: preprocess.py
# Author:zhaozhenyu
# Mail: zhaozhenyu_tx@126.com
# Created Time: 10:09:06 2018-08-31
#########################################################################
import sys
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import configparser
from config import Config
import traceback
#config
html_re=r'</?\w+[^>]*>'


def get_local_dict(path):
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

class JsonDataReader:
    def __init__(self,config,run_type):
        self.config = config
        if run_type == 'train':
            self.input_file = config.TRAIN_FILE
        elif run_type == 'dev':
            self.input_file = config.DEV_FILE
        else:
            self.input_file = config.TEST_FILE
        self.run_type = run_type
        self.stop_word_path = config.STOP_WORD
        self.token_title_dir = config.TOKENIZER_TITLE_DIR
        self.token_content_dir = config.TOKENIZER_CONTENT_DIR
        self.embedding_file = config.W2V_PATH
        self.embedding_dim = config.W2V_DIM
        self.title_len = config.TITLE_LEN
        self.content_len = config.CONTENT_LEN
        self.max_features=config.IDF_SIZE
        self.title_vocab_dir = config.TITLE_VOCAB
        self.content_vocab_dir = config.CONTENT_VOCAB

        self.get_stop_word_dict()
        if run_type == 'train':
            self.load_pretrained_embedding()
        self.title_token = self.get_tokenizer(self.token_title_dir)
        self.content_token = self.get_tokenizer(self.token_content_dir)
        self.label_dict = get_local_dict(config.LABEL_DICT)
        self.person_dict = get_local_dict(config.PERSON_DICT)


    def get_stop_word_dict(self):
        print('loading stopword...')
        stopwords = {}
        for line in open(self.stop_word_path):
            stopwords[line.strip()] = 0 
        print('total '+str(len(stopwords))+' stopwords!')
        self.stopwords = stopwords 

    def load_pretrained_embedding(self):
        embedding = {}
        print('loading pretrained embedding '+self.embedding_file+' ...')
        if self.embedding_file!='':
            for line in open(self.embedding_file):
                data = line.strip().split(' ')
                em = [float(i) for i in data[1:]]
                if len(em) != self.embedding_dim:
                    continue
                embedding[data[0]] = np.array(em)
        print('total '+str(len(embedding))+' word embedding')
        self.pretrained_embedding = embedding

    def get_tokenizer(self,token_dir):
        if self.run_type=='train':
            tokenizer = TfidfVectorizer(dtype=np.float32,min_df=5,max_features=self.max_features)
        else:
            tokenizer = pickle.load(open(token_dir,'rb'))
        return tokenizer

    def data(self):
        self.load_input_file()
        content_tfidf,content_token = self.get_tfidf_vec(
                self.content,token = self.content_token,
                sv_path = self.content_vocab_dir,token_path = self.token_content_dir)
        #self.content_token = content_token
        title_tfidf,title_token = self.get_tfidf_vec(
                self.title,token = self.title_token,
                sv_path = self.title_vocab_dir,token_path = self.token_title_dir)
        #self.title_token = titile_token

        title_pad,title_em = self.get_padded_vec(doc=self.title,token=self.title_token,lenth=self.title_len)
        content_pad,content_em = self.get_padded_vec(doc=self.content,token=self.content_token,lenth=self.content_len)
        #label
        label = self.get_label_vec()
        #static_feature
        static_feature = self.get_static_features()

        res = {
                'label':label,
                'title_pad':title_pad,
                'content_pad':content_pad,
                'title_em':title_em,
                'content_em':content_em,
                'content_tfidf':content_tfidf,
                'static_features':static_feature,
                'ori_x':self.ori_x
                }
        
        return res

    def get_tfidf_vec(self,doc,token,sv_path,token_path):
        if self.run_type =='train':
            idf = token.fit_transform(doc)#.toarray()
            pickle.dump(token,open(token_path,'wb'),0)
            vocab_list = sorted(token.vocabulary_.items(),key=lambda x:x[1],reverse=True)
            out = open(sv_path,'w')
            for item in vocab_list:
                out.write(item[0]+'\t'+str(item[1])+'\n')
            out.close()
        else:
            idf = token.transform(doc)#.toarray()
        print('now tfidf token has '+str(len(token.vocabulary_)))
        return idf ,token

    def get_padded_vec(self,doc,token,lenth):
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
        print('starting padded vec....')
        vocab = token.vocabulary_
        if self.run_type =='train':
            em = np.random.rand(len(vocab)+3,self.embedding_dim)
            for k in vocab:
                if k in self.pretrained_embedding:
                    em[vocab[k]+3] = self.pretrained_embedding[k]
        else:
            em = {}
        word_ids = texts_to_sequences(doc,vocab)
        padded_input = pad_sequences(word_ids,maxlen=lenth,padding='post',truncating='post') 
        print(str(padded_input[0]))
        return padded_input,em


    def load_input_file(self):
        title = []
        content = []
        url = []
        person = []
        c_word = []
        c_para = []
        c_pic = []
        c_title = []
        len_para = []
        ori_x = []
        label = []
        docid = []
        print('reading input file')
        count = 0
        for lines in open(self.input_file):
            count +=1
            if count%1000==1:
                print("\r"+str(count),end='')
            try:
                data = lines.strip().split('\t')
                cur_label = eval(data[0])
                new_label = self.get_label(cur_label) 
                if len(new_label)==0:
                    continue
                jd = json.loads(data[2])
                cur_title = str(jd.get('title')).lower()
                cur_content = str(jd.get('content')).lower()
                cur_url = str(jd.get('url')).lower()
                cur_person = jd.get('ne_content_person')
                cur_c_word = jd.get('c_word')
                cur_c_para = jd.get('para_count')
                cur_c_pic = jd.get('image_count')
                cur_c_title = str(jd.get('title_c_count'))
                cur_len_para = str(jd.get('para_length'))
                cur_cate = str(jd.get('text_category'))
                cur_id = str(jd.get('docid'))
                cur_ori_x = str('\t'.join([str(new_label),str(cur_label),cur_url,str(cur_id),str(cur_person),cur_title,cur_cate]))
            except Exception as e:
                print(traceback.format_exc())
                continue
            title.append(cur_title)
            content.append(cur_content)
            url.append(cur_url)
            person.append(cur_person)
            c_word.append(cur_c_word)
            c_para.append(cur_c_para)
            c_pic.append(cur_c_pic)
            c_title.append(cur_c_title)
            len_para.append(cur_len_para)
            ori_x.append(cur_ori_x)
            label.append(new_label)
        print('input number is '+str(len(title)))
        self.title = title
        self.content = content
        self.person = person
        self.c_word = c_word
        self.c_para = c_para
        self.c_pic = c_pic
        self.c_title = c_title
        self.len_para = len_para
        self.ori_x = ori_x
        self.label = label

    def get_label(self,label_list):
        label = {}
        for item in label_list:
            for i in range(len(item[2].split('_'))):
                if item[2]!='':
                    cur = '_'.join(item[2].split('_')[:(i+1)])
                    if cur in self.label_dict:
                        label[cur]  = 1
        '''
            if item[2].split('_')[0] in self.label_dict:
                label = item[2].split('_')[0]
                break
        '''
        return label

    def get_label_vec(self):
        print('get label vec '+str(self.label_dict))
        res = np.zeros((len(self.label),len(self.label_dict)))
        for i in range(len(self.label)):
            for k in self.label[i]:
                res[i][self.label_dict[k]] = 1
        print(str(res[0]))
        return res
    
    def get_num(self,num):
        res = 0.0
        try:
            res = float(num)
        except:
            pass
        return res

    def get_static_features(self):
        res = []
        print('starting static features')
        for i in range(len(self.title)):
            cur = []
            #celebrities
            cur_person = self.person[i]
            p_num = 0
            if type(cur_person)==type({}):
                for k in cur_person:
                    if k in self.person_dict:
                        p_num += cur_person[k]
            cur.append(p_num)
            #c_word
            cur.append(self.get_num(self.c_word[i]))

            #c_para
            cur.append(self.get_num(self.c_para[i]))

            #image_count
            cur.append(self.get_num(self.c_pic[i]))
            #title_c_count
            cur.append(self.get_num(self.c_title[i]))
            #para_len
            cur.append(self.get_num(self.len_para[i]))

            res.append(cur)
        print('static features done')
        print(res[0])
        return np.array(res)
                



if __name__=='__main__':
    config = Config(sys.argv[1:])
    train_reader = JsonDataReader(config,run_type='test')
    
    data = train_reader.data()
