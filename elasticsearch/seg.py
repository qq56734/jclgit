def segmentation(partition):
    import os
    import re
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    import codecs

    abspath = "data"

    # # 结巴加载用户词典
    #userDict_path = os.path.join(abspath, "ITKeywords.txt")
    #jieba.load_userdict(userDict_path)
    #
    # # 停用词文本
    stopwords_path = os.path.join(abspath, "stopwords.txt")
    def get_stopwords_list():
    #     """返回stopwords列表"""
         stopwords_list = [i.strip() for i in codecs.open(stopwords_path, encoding = 'UTF-8').readlines()]
         return stopwords_list
    # # 所有的停用词列表
    stopwords_list = get_stopwords_list()



    # 分词
    def cut_sentence(sentence):
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        seg_list = pseg.lcut(sentence)
        seg_list = [i for i in seg_list if i.word not in stopwords_list]
        filtered_words_list = []
        for seg in seg_list:
            if len(seg.word) <= 1:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    result = []
    for row in partition:
        #sentence = re.sub("<.*?>", "")  # 替换掉标签数据
        words = cut_sentence(row)
        result += words
    return result
