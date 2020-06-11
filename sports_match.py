#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: sports_match.py
# Author: NLP_zhaozhenyu
# Mail: zhaozhenyu_tx@126.com
# Created Time: 16:22:29 2020-06-08
#########################################################################
import sys
from ahocorapy.keywordtree import KeywordTree
import json

def load_sports_dict(path):
    res = {}
    for lines in open(path):
        data = lines.strip().split('\t')
        if len(data)==2:
            res[data[0].lower()] = data[1]
    return res

sports_dict = load_sports_dict('./dict/sports.exdict')
sports_kt = KeywordTree(case_insensitive=False)

for k in sports_dict.keys():
    sports_kt.add(k)
sports_kt.finalize()

def search(content):
    res = sports_kt.search_all(content.lower())
    final = {}
    if res == None:
        return {}
    for k in res:
        if k[0] in sports_dict:
            if sports_dict[k[0]] not in final:
                final[sports_dict[k[0]]] = set()
            final[sports_dict[k[0]]].add(k[0])
    return final


if __name__=='__main__':
    for lines in open(sys.argv[1]):
        data = lines.strip().split('\t')[2]
        jd = json.loads(data)
        res = search(jd['content'])
        print(str(res)+'\t'+lines.strip().split('\t')[0])

