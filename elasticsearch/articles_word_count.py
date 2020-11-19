import os
import pandas as pd
from seg import segmentation
from collections import Counter
import json
from atlassian import Confluence
from bs4 import BeautifulSoup
import re




def update_oa(words, progress):

    #if progress['oa'] > 0:
        #


    oa_dicts = []

    #读取id
    with open(r'data\oa_id.txt', "r", encoding="utf-8") as f:
        ids = f.read().split()


    for id in ids[1:]:
        oa_dict = {}
        print(id)
        #根据id读取本地文章
        path = os.path.join('data','reports',  id + '.txt')
        with open(path, "r", encoding='utf-8') as f:
            title = f.readline().split()
            article = f.read()




        #分词
        seg = segmentation(article.split())

        word_count = Counter(seg)
        #print(word_count)
        wordn = sum(word_count.values())
        for word in word_count:
            word_count[word] = word_count[word] / wordn
        #print(word_count)

        words += Counter(word_count.keys())


        oa_dict['id'] = id

        oa_dict['seg'] = word_count

        oa_dict['title'] = title

        oa_dict['text'] = article


        oa_dicts.append(oa_dict)

    oa_df = pd.DataFrame(oa_dicts)

    print(oa_df)

    oa_df.to_json('data/seg/articles_oa.json')

    progress['oa'] = len(oa_dicts)


def update_confluence(words, progress):


    confluence = Confluence(
    url='http://docs.fscut.com',
    username='jiangchenglin',
    password='jcl565600')

    #if progress['oa'] > 0:
        #


    confluence_dicts = []

    #读取id
    with open(r'data\id_confluence.txt', "r", encoding="utf-8") as f:
        ids = f.read().split()


    for id in ids[1:]:



        #根据id从接口获取页面信息
        page_info = confluence.get_page_by_id(id, expand='space,body.view,version,container')
        content = page_info['body']['view']['value']


        content = page_info['body']['view']['value']
        soup = BeautifulSoup(content, 'lxml')
        article = soup.text
        title = page_info['title']


        if len(title) < 15:
            continue


        confluence_dict = {}
        print(id)


        #根据id读取本地文章
        #path = os.path.join('data','reports',  id + '.txt')
        # with open(path, "r", encoding='utf-8') as f:
        #     title = f.readline().split()
        #     article = f.read()

        #分词
        seg = segmentation(article.split())



        print(seg)


        word_count = Counter(seg)
        #print(word_count)
        wordn = sum(word_count.values())
        for word in word_count:
            word_count[word] = word_count[word] / wordn
        #print(word_count)

        words += Counter(word_count.keys())


        confluence_dict['id'] = id

        confluence_dict['seg'] = word_count

        confluence_dict['title'] = title

        confluence_dict['text'] = article

        confluence_dict['date'] = title.split(' ')[0]
        print(confluence_dict['date'])

        #从标题解析作者
        confluence_dict['author'] = re.split('[(-]', title)[-2]

        print(confluence_dict['author'])

        confluence_dict['url'] = 'http://docs.fscut.com/pages/viewpage.action?pageId=' + id



        confluence_dicts.append(confluence_dict)




    confluence_df = pd.DataFrame(confluence_dicts)

    print(confluence_df)

    confluence_df.to_json('data/seg/articles_confluence.json')

    progress['confluence'] = len(confluence_dicts)




if __name__ == '__main__':

    #读取词频统计
    with open("data\wordcount.json") as f:
        words_count = json.load(f)

    #读取更新进度
    with open("data\progress.json") as f:
        progress = json.load(f)


    words_count = Counter(words_count)

    #更新coufluence
    #update_oa(words_count, progress)
    update_confluence(words_count, progress)


    #保存新的词频统计
    # with open("data\wordcount.json", 'w') as f:
    #     json.dump(words_count,f)

    # #保存更新进度
    # with open("data\progress.json", 'w') as f:
    #     json.dump(progress,f)








