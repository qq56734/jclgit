import json
from collections import Counter
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')
import pandas as pd


with open("data\wordcount.json") as f:
    words_count = json.load(f)

words_count = Counter(words_count)


words_count = words_count.most_common(len(words_count))


print(words_count)

# print(len(words_count))
# print(words_count)


for i in range(len(words_count)):
    updict = {'keyword': words_count[i][0], 'count': words_count[i][1], 'rank': i + 1}
    print(updict)
    es.index(index="keywords_all", body=updict)



#print(words_count)
