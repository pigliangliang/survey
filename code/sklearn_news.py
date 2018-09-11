#author_by zhuxiaoliang
#2018-08-31 下午7:28


import pandas as pd
import jieba
from gensim import models, corpora, similarities
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import jieba.posseg as psg

from sklearn.cluster import DBSCAN

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from sklearn.feature_selection import VarianceThreshold

def read_file(path=None):
    csv_data1 = pd.read_csv('../data/-1.csv')
    csv_data2 = pd.read_csv('../data/1.csv')
    csv_data3 = pd.read_csv('../data/0.csv')
    frames = [csv_data1,csv_data2,csv_data3]
    csv_data = pd.concat(frames)
    return csv_data


def load_stopwords(path='../data/cnews/stopwords.txt'):
    stopwords = [line.strip() for line in open(path, 'r').readlines()]
    return stopwords


def seg_file(contents, stopwords):
    # 分词
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
    # 词性分词
    seg = []
    for con in contents:
        local_seg = psg.cut(con)
        local = []
        for s in local_seg:
            if s.flag.startswith('d') or s.flag.startswith('a') or s.flag.startswith('v') or s.flag.startswith('n'):
                #if len(s.word) != 1:
                    # print(s.word,end=' ')
                local.append(s.word)
        seg.append(local)
    return seg

import pickle
def build_feature(documents):
    vectorizer = TfidfVectorizer(max_features=20000,ngram_range=(2,2))
    feature_matrix = vectorizer.fit_transform(documents)
    word_bag = vectorizer.get_feature_names()
    pickle.dump(word_bag, open('word_bag.pkl', 'wb'))
    return vectorizer, feature_matrix


def read_label(path='score.txt'):
    label = []
    with open(path, 'r') as f:
        label.extend(f.readlines())

    return label



if __name__ =="__main__":

    #读取内容


    contents_pd = read_file()
    print(len(contents_pd[0].tolist()))
    contents = contents_pd[0].tolist()
    labels = contents_pd[1].tolist()
    print(len(labels))

    #分词，去除停用词处理
    stopword = load_stopwords()

    seg = seg_file(contents,stopword)
    print(seg[0])

    #提取特征

    vector,feature_matrix = build_feature([' '.join(doc) for doc in seg])
    '''
    # 移除方差较低的特种特征
    sel = VarianceThreshold()
    feature_matrix = sel.fit_transform(feature_matrix)
    print(feature_matrix.shape)'''

    #label = read_label()
    x_train,x_test,y_train,y_test = train_test_split(feature_matrix,labels,test_size=0.25,random_state=33)
    print(x_train.shape,y_train.shape)


    #



    #train
    lr = LogisticRegression()
    lr.fit(x_train,y_train)

    print(lr.score(x_test,y_test))

    y_pred = lr.predict(x_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(value=lr, filename='lr.gz', compress=True)
    print('model saved! ')



    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB,GaussianNB
    from sklearn.model_selection import cross_val_score

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    count = 1
    for model in models:

        model.fit(x_train,y_train)
        print(model.score(x_test, y_test))

        y_pred = model.predict(x_test)
        print(classification_report(y_test, y_pred))
        joblib.dump(value=model, filename='{}.gz'.format(count), compress=True)
        print('model saved! ')
        count+=1

    #
    #text = '"经济观察网记者李紫宸 8月29日上午，中华全国工商业联合会在辽宁沈阳发布2018中国民营企业500强榜单。过去一年中（2017年），民营企业500强入围门槛为156.84亿元，500强营收总额为244793.82亿元，此前一年这一数字为193616.14亿元。其中，营收总额超过1000亿元的有42家，上一年为27家，500到1000亿之间的则有91家，上一年为64家。2017年，民营企业500强产业结构中第二产业仍占主体地位，但入围企业数量有所减少。从入围企业数量来看，民营企业500强仍以制造业为主导，钢铁行业仍居前列，但数量较上一年度有所增加，综合业首次成为第二大主体，民营企业500强行业结构进一步优化。从资产规模来看，民营企业500强产业结构延续往年态势，第二产业资产规模占比继续降低，第三产业资产规模占比持续上升。各行业经营效益整体有所提升，煤炭开采和洗选业，医药制造业，铁路、船舶、航空航天和房地产业的销售净利率、资产净利率和净资产收益率相对较高；货币金融服务业，畜牧业等行业经营效益较上年明显下降。具体来看，本年度民营企业500强企业呈现以下主要特点：一、服务业比重增强，产业结构优化升级'
    text = '人民币汇率下跌'
    model = joblib.load('2.gz')
    #print(model)
    # 加载向量化模型
    word_bag = pickle.load(open('word_bag.pkl', 'rb'))
    v = TfidfVectorizer(vocabulary=word_bag, token_pattern=r'\w+')
    #seg = jieba.cut(text)
    import  jieba.posseg as psg
    seg = psg.cut(text)
    local = []
    for s  in seg:
        if s.flag.startswith('d') or s.flag.startswith('a') or s.flag.startswith('v') or s.flag.startswith('n'):
            local.append(s.word)

    print(' '.join(local))
    mytest = v.fit_transform([' '.join(local)])
    print(mytest.shape)
    print(model.predict(mytest))
