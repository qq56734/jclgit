from atlassian import Jira
from atlassian import Confluence
from atlassian import Bitbucket
from atlassian import ServiceDesk

confluence = Confluence(
    url='http://docs.fscut.com',
    username='jiangchenglin',
    password='jcl565600')

ids = []

for i in range(50):
    print(i)

    page_dicts = confluence.get_all_pages_from_space('~oa', start=i*100, limit=(i+1)*100, status=None, expand='body', content_type='page')
    for page_dict in page_dicts:
        ids.append(page_dict['id'])

with open(r'data/id_confluence.txt', "w", encoding="utf-8") as f:
    for id in ids:
        f.write(str(id)+"\n")