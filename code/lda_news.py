#author_by zhuxiaoliang
#2018-08-29 下午2:07

import pandas as pd
import jieba
from gensim import models,corpora,similarities
import jieba.posseg as psg

def read_file(path=None):

    csv_data = pd.read_csv('../data/news_crawl_temp.csv',header=None,index_col=None)

    return  csv_data

def load_stopwords(path='../data/cnews/stopwords.txt'):

    stopwords = [line.strip() for line in open(path, 'r').readlines()]
    return stopwords

def seg_file(contents,stopwords):
    '''
    seg = []

    for con in contents:
        local_seg = jieba.cut(con,False)
        local = []
        for word in local_seg:
            if word not in stopwords and word.strip() !='':
                local.append(word)
        seg.append(local)
    return seg'''
    seg = []
    for con in contents:
        local_seg = psg.cut(con)
        local = []
        for s in local_seg:
            if s.flag.startswith('d') or s.flag.startswith('a') or s.flag.startswith('v') or s.flag.startswith('n'):
                if len(s.word) != 1:
                    # print(s.word,end=' ')
                    local.append(s.word)
        seg.append(local)
    return seg



if __name__ =='__main__':

    #数据处理
    contents = read_file()
    stopwords = load_stopwords()

    seg = seg_file(contents[0],stopwords)
    # 建立字典
    dictionary = corpora.Dictionary(seg)
    V = len(dictionary)
    print(V)

    # 统计文档词频矩阵
    text = [dictionary.doc2bow(text,allow_update=True) for text in seg]
    #print(text[0])#稀疏矩阵

    #计算Tfidf矩阵
    text_tfidf = models.TfidfModel(text)[text]

    #建立LDA模型,输出前十个主题
    lda = models.LdaModel(text_tfidf,id2word=dictionary,num_topics=200,iterations=100)


    #显示主题
    for k,v in lda.print_topics(num_topics=10):
         print(k,v)
    #所有文档的主题
    doc_topic = lda.get_document_topics(text_tfidf)
    print(len(doc_topic))
    for dt in doc_topic:
        print(dt)
        d = dict(dt)
        ret = sorted(d.items(),key=lambda x:x[1],reverse=True)[0]
        print(ret[0])
        for k,v in lda.print_topics(num_topics=200):
            if k==ret[0]:
             print(k,v)
        #print(lda[text_tfidf][0])
