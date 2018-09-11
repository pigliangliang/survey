#author_by zhuxiaoliang
#2018-08-30 上午11:06


import pandas as pd
import jieba
from gensim import models,corpora,similarities
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
import jieba.posseg as psg

from sklearn.cluster import  DBSCAN

def read_file(path=None):

    csv_data = pd.read_csv('../data/news_crawl_temp.csv',header=None,index_col=None)

    return  csv_data

def load_stopwords(path='../data/cnews/stopwords.txt'):

    stopwords = [line.strip() for line in open(path, 'r').readlines()]
    return stopwords

def seg_file(contents,stopwords):
    #分词
    """
    seg = []

    for con in contents:
        local_seg = jieba.cut(con,False)
        local = []
        for word in local_seg:
            if word not in stopwords and word.strip() !='':
                local.append(word)
        seg.append(local)
    return seg"""
    #词性分词
    seg=[]
    for con in contents:
        local_seg = psg.cut(con)
        local = []
        for s in local_seg:
            if  s.flag.startswith('d') or s.flag.startswith('a') or s.flag.startswith('v') or s.flag.startswith('n'):
                if len(s.word) != 1:
                    # print(s.word,end=' ')
                    local.append(s.word)
        seg.append(local)
    return seg

def build_feature(documents):

    vectorizer = TfidfVectorizer()

    feature_matrix = vectorizer.fit_transform(documents)

    return vectorizer,feature_matrix

def read_label(path='score.txt'):
    label = []
    with open(path,'r') as f:
        label.append(f.readlines().split('\n'))

    return label




def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def get_cluster_data(clustering_obj, content_pd,
                     feature_names, num_clusters,
                     topn_features=50):
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征

    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        contents = content_pd[content_pd['cluster'] == cluster_num].values.tolist()
        cluster_details[cluster_num]['content'] = contents

    return cluster_details


def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 50)
        print('Key features:', cluster_details['key_features'])
        print('content in this cluster:')
        for cd in cluster_details['content']:
            #截取开头部分内容
            print(cd[0][:30])
            print('#'*30)
        print('=' * 50)



if __name__ =="__main__":

    #读取内容

    contents_pd = read_file()
    print(len(contents_pd[0].tolist()))
    contents = contents_pd[0].tolist()

    #分词，去除停用词处理
    stopword = load_stopwords()

    seg = seg_file(contents,stopword)
    #print(seg[0])

    #提取特征

    vector,feature_matrix = build_feature([' '.join(doc) for doc in seg])

    print(feature_matrix.shape)
    feature_names = vector.get_feature_names()

    #print(feature_names[:10])

    #聚类

    number_cluster = 5
    km_obj ,clusters = k_means(feature_matrix,number_cluster)
    print(clusters)
    contents_pd['cluster']=clusters

    from collections import Counter
    c = Counter(clusters)
    print('获取每个簇的分类数量',sorted(c.items(),key=lambda x:x[1],reverse=True))

    cluster_data = get_cluster_data(km_obj,contents_pd,feature_names,number_cluster)
    print_cluster_data(cluster_data)


