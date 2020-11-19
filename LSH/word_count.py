from time import sleep

import pandas as pd
from sqlalchemy import create_engine, Integer, VARCHAR
from util import consts
from seg import segmentation
import json
from nohtml import strip_tags
from collections import Counter

from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')
from pandasticsearch import Select
import numpy as np




def update_story(words, progress , updaterate):


    startid = progress['story']


    if startid > 0:
        story_p = pd.read_json('data/pms_story.json')
    else :
        story_p = pd.DataFrame()





    while True:

        #sleep(1)

        result_dict = es.search(index="pms_story", body={
            "query": {
                "constant_score": {
                    "filter": {
                        "range": {
                            "story": {
                                "gt": startid,
                                "lt": startid + updaterate + 1
                            }
                        }
                    }
                }
            }
        }, size=10000)

        if len(result_dict['hits']['hits'] ) == 0:
            break

        story_new = Select.from_dict(result_dict).to_pandas()[['story', 'title', 'spec']]


        # sql = 'select story, title, spec from zt_storyspec where story > {0} limit {1}'.format(startid, updaterate)
        # story_new = pd.read_sql(sql, engine)
        # ifremain = (len(story_new) == updaterate)
        # print(len(story_new))
        #
        # print(ifremain)


        # from nohtml import strip_tags
        # story_new['spec'] = story_new['spec'].apply(lambda x:strip_tags(x))
        story_new = story_new.loc[story_new['story'].drop_duplicates().index,:]

        story_new = story_new.set_index(story_new['story'].values)





        story_new['seg'] = ''
        for i in story_new['story'].values:

            seg = segmentation((story_new['title'][i] + story_new['spec'][i]).split())
            word_count = Counter(seg)
            print(i)
            print(word_count)
            wordn = sum(word_count.values())
            for word in word_count:
                word_count[word] = word_count[word] / wordn
            #print(word_count)

            words += Counter(word_count.keys())
            story_new['seg'][i] = dict(word_count)

        story_p = pd.concat([story_p, story_new], axis = 0)



        startid = np.sort(story_new['story'].values)[-1]

        print(startid)






    progress['story'] = int(startid)

    story_p.to_json('data/pms_story.json')


def update_bug(words, progress, updaterate):

    startid = progress['bug']


    if startid > 0:
        bug_p = pd.read_json('data/pms_bug.json')
    else :
        bug_p = pd.DataFrame()

    while True:

        # sleep(1)

        result_dict = es.search(index="pms_bug", body={
            "query": {
                "constant_score": {
                    "filter": {
                        "range": {
                            "id": {
                                "gt": startid,
                                "lt": startid + updaterate + 1
                            }
                        }
                    }
                }
            }
        }, size=10000)

        if len(result_dict['hits']['hits']) == 0:
            break

        bug_new = Select.from_dict(result_dict).to_pandas()[['id', 'title', 'steps']]

        bug_new = bug_new.loc[bug_new['id'].drop_duplicates().index,:]

        bug_new = bug_new.set_index(bug_new['id'].values)





        bug_new['seg'] = ''
        for i in bug_new['id'].values:
            #print(i)


            seg = segmentation((bug_new['title'][i] + bug_new['steps'][i]).split())
            word_count = Counter(seg)
            #print(word_count)
            wordn = sum(word_count.values())
            for word in word_count:
                word_count[word] = word_count[word] / wordn
            #print(word_count)

            words += Counter(word_count.keys())
            bug_new['seg'][i] = dict(word_count)

        bug_p = pd.concat([bug_p, bug_new], axis = 0)

        startid = np.sort(bug_new['id'].values)[-1]
        print(startid)



    progress['bug'] = int(startid)

    bug_p.to_json('data/pms_bug.json')

def update_task(words, progress, updaterate):


    startid = progress['task']


    if startid > 0:
        task_p = pd.read_json('data/pms_task.json')
    else :
        task_p = pd.DataFrame()

    while True:

        # sleep(1)

        result_dict = es.search(index="pms_task", body={
            "query": {
                "constant_score": {
                    "filter": {
                        "range": {
                            "id": {
                                "gt": startid,
                                "lt": startid + updaterate + 1
                            }
                        }
                    }
                }
            }
        }, size=10000)

        if len(result_dict['hits']['hits']) == 0:
            break

        task_new = Select.from_dict(result_dict).to_pandas()[['name', 'id', 'desc']]

        # from nohtml import strip_tags
        # task_new['desc'] = task_new['desc'].apply(lambda x:strip_tags(x))
        task_new = task_new.loc[task_new['id'].drop_duplicates().index,:]

        task_new = task_new.set_index(task_new['id'].values)


        #task_new['id'] = task_new['id'].astype(str)

        task_new['seg'] = ''

        for i in task_new['id'].values:
            print(i)

            seg = segmentation((task_new['name'][i] + task_new['desc'][i]).split())
            word_count = Counter(seg)
            print(word_count)
            wordn = sum(word_count.values())
            for word in word_count:
                word_count[word] = word_count[word] / wordn
            print(word_count)

            words += Counter(word_count.keys())
            task_new['seg'][i] = dict(word_count)

        task_p = pd.concat([task_p, task_new], axis = 0)

        startid = np.sort(task_new['id'].values)[-1]

    progress['task'] = int(startid)


    task_p.to_json('data/pms_task.json')



 if __name__ == '__main__':

    update_rate = 1000

    with open("data\wordcount.json") as f:
        words_count = json.load(f)

    with open("data\progress.json") as f:
        progress = json.load(f)

    words_count = Counter(words_count)

    update_story(words_count, progress, update_rate)
    update_bug(words_count, progress, update_rate)
    update_task(words_count, progress, update_rate)


    with open("data\wordcount.json", 'w') as f:
        json.dump(words_count,f)


    with open("data\progress.json", 'w') as f:
        json.dump(progress,f)
