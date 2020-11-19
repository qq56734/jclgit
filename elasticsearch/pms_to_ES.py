from datetime import datetime
from elasticsearch import Elasticsearch
import math
import pandas as pd
from collections import Counter
es = Elasticsearch('127.0.0.1:9200')
import json


def story_to_es(total):

    story_df = pd.read_json('data/seg/pms_story.json')
    story_df['keywords'] = ''

    for i in story_df['story']:
        tf = Counter(story_df['seg'][i])
        for word in tf:
            tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
        tf = tf.most_common(10)
        print(tf)
        story_df['keywords'][i] = [w[0] for w in tf]
        print(i)
        es.index(index="pms_story_keywords", id=story_df['story'][i], body=dict(story_df.loc[i, ['story', 'title', 'spec', 'keywords']]))

def task_to_es(total):

    task_df = pd.read_json('data/seg/pms_task.json')
    task_df['keywords'] = ''

    for i in task_df['id']:
        tf = Counter(task_df['seg'][i])
        for word in tf:
            tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
        tf = tf.most_common(10)
        print(tf)
        task_df['keywords'][i] = [w[0] for w in tf]
        print(i)
        es.index(index="pms_task_keywords", id=task_df['id'][i], body=dict(task_df.loc[i, ['id', 'name', 'desc', 'keywords']]))


def bug_to_es(total):

    bug_df = pd.read_json('data/seg/pms_bug.json')
    bug_df['keywords'] = ''

    for i in bug_df['id']:
        tf = Counter(bug_df['seg'][i])
        for word in tf:
            tf[word] = round(tf[word] * math.log(total / (words_count[word] + 1)), 2)
        tf = tf.most_common(10)
        print(tf)
        bug_df['keywords'][i] = [w[0] for w in tf]
        print(i)
        es.index(index="pms_bug_keywords", id=bug_df['id'][i], body=dict(bug_df.loc[i, ['id', 'title', 'steps', 'keywords']]))

global words_count

if __name__ == '__main__':


    with open("data\wordcount.json") as f:
        words_count = Counter(json.load(f))

    with open("data\progress.json") as f:
        progress = json.load(f)


    total = progress['story'] + progress['bug'] + progress['task'] + progress['oa'] + progress['confluence']

    story_to_es(total)
    task_to_es(total)
    bug_to_es(total)

