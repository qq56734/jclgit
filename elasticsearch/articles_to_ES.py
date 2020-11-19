from elasticsearch import Elasticsearch
from seg import segmentation
from atlassian import Confluence
import requests
import json
import os
import re
from collections import Counter
import math
import pandas as pd




def get_oa_date(id, page_dict):
    for page_info in page_dict:
        if page_info['id'] == id:
            return [page_info['issueTime'], page_info['issueName']]

# def oa_to_es(total):
#
#
#     s = requests.Session()
#     url = 'http://oa.fscut.com/seeyon/ajax.do?d=4095&method=ajaxAction&managerName=bbsArticleManager'
#
#     cookie = 'JSESSIONID=D8C537CF85FCF3FCD78ED738491279D9'
#
#     headers = {
#                 'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36',
#                 'cookie': cookie,
#                 'Referer': 'http://oa.fscut.com/seeyon/bbs.do?method=bbsIndex&portalId=-3489196460133432033&_resourceCode=F05_bbsIndexAccount'
#             }
#     formdata = {
#                     "managerMethod": "findListData",
#                     "arguments":'[{"pageSize":"3000","pageNo":1,"spaceType":"","spaceId":"","boardId":"","listType":"latestArticle","condition":"","textfield1":"","textfield2":""}]'
#                }
#
#
#     r = s.post(url, data = formdata, headers = headers)
#
#
#     oa_page_dict = json.loads(r.text)
#     oa_page_dict = oa_page_dict['list']
#
#     print(oa_page_dict)
#
#     oa_df = pd.read_json('data/seg/articles_oa.json')

#
#
#
#
#     for i in range(len(oa_df)):
#         doc = dict(oa_df.loc[i,:])
#         id = str(doc['id'])
#
#         print(id)
#
#         doc['url'] = 'http://oa.fscut.com/seeyon/bbs.do?method=bbsView&articleId=' + id + '&spaceType=&spaceId='
#         doc['issue_date'] = get_oa_date(id, oa_page_dict)[0][0:10]
#         doc['author'] = get_oa_date(id, oa_page_dict)[1]
#
#         tf = Counter(doc['seg'])
#         for word in tf:
#             tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
#         tf = tf.most_common(10)
#         print(tf)
#         doc['keywords'] = [w[0] for w in tf]
#
#
#         es.index(index="articles_test", id=id, body=doc)



    # for i in range(len(oa_df) + 1):
    #     tf = Counter(bug_df['seg'][i])
    #     for word in tf:
    #         tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
    #     tf = tf.most_common(10)
    #     print(tf)
    #     bug_df['keywords'][i] = [w[0] for w in tf]
    #     print(i)
    #     es.index(index="pms_bug_keywords", id=bug_df['id'][i], body=dict(bug_df.loc[i, ['id', 'title', 'steps', 'keywords']]))



def get_confluence_date(id):
    page_info = confluence.get_page_by_id(id, expand = 'space,body.view,version,container')
    return page_info['version']['when']



def update_confluence(total):

    confluence_df = pd.read_json('data/seg/articles_confluence.json')


    for i in range(len(confluence_df)):
        doc = dict(confluence_df.loc[i,:])
        id = doc['id']

        print(id)
        print(i)

        tf = Counter(doc['seg'])
        for word in tf:
            tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
        tf = tf.most_common(10)
        print(tf)
        doc['keywords'] = [w[0] for w in tf]
        doc.pop('seg')
        if '/' in doc['date'] or '.' in doc['date']:
            #dates = doc['date'].split('/')
            dates =  re.split('[./-]', doc['date'])[0:3]
            print(dates)
            for i in range(1,3):
                if len(dates[i]) == 1:
                    dates[i] = '0' + dates[i]
            doc['date'] = '-'.join(dates[0:3])







        print(doc['date'])


        es.index(index="articles_test", id=id, body=doc)



if __name__ == '__main__':


    with open("data\wordcount.json") as f:
        words_count = Counter(json.load(f))

    #confluence 请求信息
    confluence = Confluence(
        url='http://docs.fscut.com',
        username='jiangchenglin',
        password='jcl565600')



    es = Elasticsearch('127.0.0.1:9200')

    with open("data\progress.json") as f:
        progress = json.load(f)


    total = progress['story'] + progress['bug'] + progress['task']  + progress['confluence']


    update_confluence(total)









