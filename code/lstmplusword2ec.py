#author_by zhuxiaoliang
#2018-08-22 上午9:51
#w2v+lstm二分类
#参考网址 https://buptldy.github.io/2016/07/20/2016-07-20-sentiment%20analysis/


import yaml
import pandas as pd
import numpy as np
import jieba

from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec,LineSentence
from gensim.corpora import Dictionary
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense,Dropout,Activation
from keras.models import model_from_yaml
#np.random.seed(33)#随机种子


#超参数

batch_size = 32
n_epoch = 4
input_length = 100


#加载文件
def load_file():
    neg = pd.read_excel('../data/neg.xls',header=None,index=None)
    pos = pd.read_excel('../data/pos.xls',header=None,index=None)
    posplusneg = np.concatenate((pos[0],neg[0]))
    label = np.concatenate((np.ones(len(pos),dtype=int),np.zeros(len(neg),dtype=int)))
    return posplusneg,label
#分词，去掉换行符

def tokenzier(document):
    text = [jieba.lcut(text.replace('\n','')) for text in document]
    return text

def create_dictionaries(model=None,conbined=None):
    if (model is not None) and (conbined is not None):
        gensim_dic = Dictionary()
        gensim_dic.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2index = {v:k+1 for k,v in gensim_dic.items()}
        w2vec = {word:model[word] for word in w2index.keys()}

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2index[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        conbined = parse_dataset(conbined)
        #输入要求长度一致，所以句子要截取同样长度，不足最大长度补零
        conbined = sequence.pad_sequences(conbined,maxlen=100)
        return w2index,w2vec,conbined
    else:
        print('data is None')


def word2vec_train(document):
    #训练方式一
    model = Word2Vec(size=100,min_count=5,window=5)

    model.build_vocab(document)
    model.train(document,total_examples=15828,epochs=1)
    model.save('../data/lstm_word2v_model.pkl')
    """
    #训练方式二读取文本数据
    model = Word2Vec(LineSentence(document),size=100,window=5,min_count=5)
    model.save('../data/lstm_word2v_model.pkl')
    """
    index_dict, word_vectors, conbined = create_dictionaries(model=model, conbined=document)
    return index_dict, word_vectors, conbined

def get_data(index_dic,word2v,conbined,label):
    n_symbols = len(index_dic)+1#所有的单词索引，从零开始，所以加1
    #嵌入层维度
    embedding_weigths = np.zeros((n_symbols,100))#100是前文设置的每个词向量的维度。共计n_symbols 个词向量。所以嵌入层维度是...
    #索引为零的单词，也就是在训练词向量的时候count次数小于5的单词词向量全都是0
    #从第一个索引单词开始查找词向量,构建嵌入层
    for word ,index in index_dic.items():
        embedding_weigths[index, :] = word2v[word]
    x_train,x_test,y_train,y_test = train_test_split(conbined,label,test_size=0.25)
    #print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weigths,x_train,y_train,x_test,y_test

#定义lstm网络结构，keras实现
def train_lstm(n_symbols,embedding_weigths,x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Embedding(output_dim=100,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weigths],
                        input_length=100))
    model.add(LSTM(output_dim=50,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print('compiling the model')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #trainning
    print('trainning')
    #print(x_train[:5],y_train[:5])
    #x_train = np.array(list(x_train))
    #y_train = np.array(list(y_train))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))


    #eval
    score = model.evaluate(x_test,y_test,batch_size=batch_size)

    #save
    yaml_string=model.to_yaml()
    with open('../data/lstm.yml','w') as f:
        f.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save_weights('../data/lstm.h5')
    print('test score:' ,score)

def train():
    conbined ,y = load_file()
    conbined = tokenzier(conbined)
    index_dict, word_vectors, conbined = word2vec_train(conbined)
    n_symbols, embedding_weights, x_train, y_train,x_test, y_test = get_data(index_dict, word_vectors, conbined, y)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


#测试
def input_transform(string):
    words = jieba.lcut(string)
    print(words)
    words = np.array(words).reshape(1,-1)
    model = Word2Vec.load('../data/lstm_word2v_model.pkl')
    _,_,conbined = create_dictionaries(model,words)
    return conbined


def lstm_predict(string):
    #加载lstm网络
    print('load model')
    with open('../data/lstm.yml','r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #
    print('load weights')
    model.load_weights('../data/lstm.h5')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)

    result = model.predict_classes(data)
    print(result[0])





if __name__=='__main__':
    #train()

    string = '我没有不开心'
    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #string = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    #string = '酒店的环境非常好，价格也便宜，值得推荐'
    #string = '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    #string = '我是傻逼'
    #string = '你是傻逼'
    #string = '屏幕较差，拍照也很粗糙。'
    #string = '质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    #string = '东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'

    lstm_predict(string)
