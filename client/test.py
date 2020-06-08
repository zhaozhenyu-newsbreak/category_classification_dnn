#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: test.py
# Author: NLP_zhaozhenyu
# Mail: zhaozhenyu_tx@126.com
# Created Time: 15:28:36 2019-01-25
#########################################################################
import sys
import requests
import json
import time

url_prefix = 'http://0.0.0.0:9111/api/v0/category_classification_dnn'
#url_prefix = 'http://172.31.128.129:9111/api/v0/category_classification_dnn'
#url_prefix = 'http://text-category-dnn.default.svc.k8sc1.nb.com:9111/api/v0/category_classification_dnn'
#url_prefix = 'http://text-category-dnn.ha.nb.com:9111/api/v0/category_classification_dnn'
test_file = sys.argv[1]

start = time.time()
max_t = 0
for lines in open(test_file):
    cur_start = time.time()
    data = lines.strip().split('\t')
    docinfo = json.loads(data[2])
    docinfo['seg_title']=docinfo['title']
    docinfo['seg_content'] = docinfo['content']
    return_info = requests.post(url_prefix,json = docinfo)
    return_dict = json.loads(return_info.text)
    cur = time.time()
    max_t = max(max_t,cur-cur_start)
    print(str(return_dict['text_category_v2'])+'\t'+json.dumps(return_dict)+'\t'+str(data[0])+'\t'+str(lines.strip()))

print(time.time()-start)
print(max_t)
