#author_by zhuxiaoliang
#2018-08-21 下午5:17

import jieba
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.svm import SVC
from gensim.models import word2vec,Word2Vec

from sklearn.linear_model import LogisticRegression


#基于svm+词向量训练
def loadfile():
    neg=pd.read_excel('../data/neg.xls',header=None,index=None)
    pos=pd.read_excel('../data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    #print(pos['words'])
    neg['words'] = neg[0].apply(cw)

    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    np.save('../data/y_train.npy',y_train)
    np.save('../data/y_test.npy',y_test)
    return x_train,x_test

#因为得到的是每个分词的向量，输入传统机器学习模型fit（x,y)中，其中x是一个完整的向量
#故将每一条评论所有的词向量加和求平均
#不如一条评论包含10个分词，那么该条评论处理后的向量是 10*词向量维度，这里是300
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

#计算词向量
def get_train_vecs(x_train,x_test):

    n_dim = 300
    w2c = Word2Vec(size=n_dim,min_count=5)
    w2c.build_vocab(x_train)
    #print(x_train)
    print('训练train向量...')
    w2c.train(x_train,total_examples=16884,epochs=3)
    #对训练集进项向量处理
    train_vec = np.concatenate([buildWordVector(z,n_dim,w2c) for z in x_train])
    np.save('../data/train_vecs.npy', train_vec)
    #对测试集做同样的处理
    print('测试集向量...')
    w2c.train(x_test,epochs=3,total_examples=4221)
    w2c.save('../data/w2v_model.pkl')
    test_vec = np.concatenate([buildWordVector(z, n_dim, w2c) for z in x_test])
    np.save('../data/test_vecs.npy', test_vec)

def get_data():
    train_vecs=np.load('../data/train_vecs.npy')
    y_train=np.load('../data/y_train.npy')
    test_vecs=np.load('../data/test_vecs.npy')
    y_test=np.load('../data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test

#svm 模型
def svm_train(x_train,y_train,x_test,y_test):
    '''

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    '''
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(x_train,y_train)
    joblib.dump(clf, '../data/svm_model.pkl')
    print(clf.score(x_test,y_test))
    '''
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print(lr.score(x_test,y_test))
    '''

##得到待预测单个句子的词向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('../data/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs


def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('../data/svm_model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print('pos')
    else:
        print('neg')



if __name__ == '__main__':
    #训练
    '''
    x_train,x_test= loadfile()
    print(len(x_train),len(x_test))
    get_train_vecs(x_train,x_test)
    train_vecs,y_train,test_vecs,y_test = get_data()
    svm_train(train_vecs,y_train,test_vecs,y_test)
    '''
    text = '好好好'
    svm_predict(text)

