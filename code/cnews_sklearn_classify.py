#author_by zhuxiaoliang
#2018-08-24 上午10:16

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import jieba
import jieba_fast
import pickle

import os



stopwords_path =  '../data/cnews/stopwords.txt'

class WordCut:
    def __init__(self,stopwords_path=stopwords_path):

        self.stopwords_path = stopwords_path
    '''
    def addDictionary(self,dict_list):
        """
        添加用户自定义字典
        :param dict_list:
        :return:
        """
        map(lambda x:jieba.load_userdict(x),dict_list)
    '''

    def seg_sentence(self,sentence,stopwords_path=None):
        """
        句子分词
        :param sentence:
        :param stopwords_path:
        :return:
        """
        if stopwords_path is None:
            stopwords_path = self.stopwords_path
        stopwords = [line.strip() for line in open(stopwords_path,'r').readlines()]
        sentence_seg = jieba.cut(sentence.strip())

        final_seg = []
        for word in sentence_seg:
            if word not in stopwords and word !=' ':
                final_seg.append(word)

        return final_seg
    def seg_file(self,path):
        """
        对文件中文本分词
        :param path:
        :return:
        """
        file_seg = []
        with open(path,'r') as f:
            for line in f.readlines():
                line_seg = self.seg_sentence(line)
                file_seg.append(line_seg)


        return  file_seg

    def deal_text(self,path):
        """
        因为文本比较特殊，已经在文档中生成【label content】格式的内容，所有在
        读出的文件中进行处理
        :param path:
        :return:
        """
        label =[]
        content =[]
        #seg_content = []
        import time
        with open(path,'r') as f:
            for line in f.readlines():

                line = line.strip()
                label.append(line.split('\t')[0])
                #start = time.time()

                content.append(' '.join(self.seg_sentence(line.split('\t')[1])))
                #end = time.time()
                #print('time',end -start)

        """
        此是已经将样本中的内容进行分开了，转化成lable部分和content部分，分别处理
        对label转化为
        label部分是中文字符，可以使用sklearn库中preprocessing包中方法处理。
        
        #label转化
        encoder = LabelEncoder()
        label = encoder.fit_transform(label)
        转化回来可以使用inverse_transform(x) x为整型常量，在预测的时候
        需要用到
        """
        """"
        seg_content = []
        #对content分词处理
        for con in content:
            seg_sentence = self.seg_sentence(con)
            seg_content.append(' '.join(seg_sentence))"""
        return label,content

def deal_sample2vector(label,content):

    # label转化
    encoder = LabelEncoder()
    label = encoder.fit_transform(label)
    for i in range(10):
        with open('labelid2string.txt','w') as f:
            f.write(encoder.inverse_transform(i)+'\n')
    #split
    x_train,x_test,y_train,y_test = train_test_split(content,label,random_state=33)
    # content 转化

    v = TfidfVectorizer(token_pattern=r'\w+')
    v.fit(x_train)
    word_bag = v.get_feature_names()
    #保存向量模型
    x_train = v.transform(x_train)
    x_test = v.transform(x_test)
    pickle.dump(word_bag,open('word_bag.pkl','wb'))
    pickle.dump(v, open('vector.pkl', 'wb'))
    return x_train,x_test,y_train,y_test


def train(x_train,y_train,x_test,y_test):
    #lr
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    print(lr.score(x_test,y_test))
    y_pred = lr.predict(x_test)
    print(classification_report(y_test,y_pred))
    joblib.dump(value=lr, filename='lr.gz', compress=True)
    print('model saved! ')
    #RF
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(n_estimators=200, random_state=1080)
    rf_model.fit(x_train, y_train)
    print("val mean accuracy: {0}".format(rf_model.score(x_test,y_test)))
    y_pred = rf_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(value=rf_model, filename='rf.gz', compress=True)
    print('model saved! ')

    '''

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]

    for model in models:
        #model_name = model.__class__.__name__
        cross_val_score(model, x_train, y_train, scoring='accuracy', cv=5)

    '''

def corpus_predict(text):
    #加载
    import json
    model = joblib.load('lr.gz')
    #加载向量化模型
    #v = pickle.load(open('vector.pkl','rb'))
    word_bag = pickle.load(open('word_bag.pkl','rb'))
    v = TfidfVectorizer(vocabulary=word_bag,token_pattern=r'\w+')
    seg = jieba.cut(text)
    mytest = v.fit_transform([' '.join(seg)])
    print(mytest.shape)
    print(model.predict(mytest))





if __name__ =='__main__':

    #train
    '''
    
    path_base ='../data/cnews/'
    wc = WordCut()

    #发现在读取文件，处理分词的时候，速度特别慢，决定用多线程处理
    
    print('deal  test contents label ')
    test_label,test_content = wc.deal_text(path_base+'cnews.test.txt')
    print('deal train')
    train_label,train_content = wc.deal_text(path_base+'cnews.train.txt')
    print('deal val')
    val_label,val_content = wc.deal_text(path_base+'cnews.val.txt')
    print(len(train_content),len(train_label))
    """
    import multiprocessing
    pool = multiprocessing.Pool(4)
    file_path = [path_base+'cnews.test.txt',path_base+'/cnews.train.txt',path_base+'/cnews.val.txt']
    for fp in file_path:
        pool.apply_async(wc.deal_text,(fp,))

        #print(len(label),len(content))
    pool.close()
    pool.join()"""
    print('deal down')
    


    content = train_content+test_content+val_content
    label =  train_label+test_label+val_label
    print('content,label len：',len(content),len(label))
    print('deal sample to vector')
    x_train,x_test,y_train,y_test = deal_sample2vector(label,content)
    print('trainning')



    train(x_train,y_train,x_test,y_test)
    
    '''
    #预测
    text = '中国基金报记者泰勒天逸一家股价不足1元的“仙股”，一家国人熟知的饮料巨头，一份昨晚公告的重组协议，一只开盘涨停的股票，一份今早的声明“直接打脸”，还有一份深交所，公司股票若连续20个交易日（不含公司股票全天停牌的交易日）收盘价均低于股票面值，将存在被强制终止上市的可能。'
    corpus_predict(text)