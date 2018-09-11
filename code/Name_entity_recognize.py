#author_by zhuxiaoliang
#2018-09-04 下午3:03


#基于BIlistm+crf 实现命名实体识别

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import numpy as np
import collections
import os

import  pickle
from  gensim.models import Word2Vec
from gensim.corpora import Dictionary
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense,Dropout,Activation
from keras.models import model_from_yaml


#读取 语料库，构建元祖（【文本】，【标记】）的数据
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path) as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    #print(data[0])
    return data


#将样本转化成字典：字：索引的形式
def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    #print(low_freq_words)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    print(word2id)
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


#读字典

def read_dictionary(vocab_path=None):
    """
    :param vocab_path:
    :return:
    """
    with open('../zh-NER-TF/data_path/wore2id_new.pkl','rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

#将句子转化为id形式
def sentence2id(sentence):
    word2id = read_dictionary()
    sentence_id = []
    for word in sentence:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def train_word2vec(data):
    # 训练方式一
    model = Word2Vec(size=100, min_count=5, window=5)

    model.build_vocab(data)
    model.train(data, total_examples=len(data), epochs=2)
    model.save('../zh-NER-TF/data_path/word2v_model.pkl')

def create_dictionaries(model=None,conbined=None):
    if (model is not None) and (conbined is not None):
        gensim_dic = Dictionary()
        gensim_dic.doc2bow(model.wv.vocab.keys(),allow_update=True)

        w2index = {v:k+1 for k,v in gensim_dic.items()}#单词 索引
        w2vec = {word:model[word] for word in w2index.keys()}#单词 向量

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




#tag转化

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }



#定义BiLSTM-CRF网络

class NER_net(object):
    BATCH_SIZE =64
    unit_num = 100  # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
    time_step = 100  # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
    DROPOUT_RATE =0.001
    EPOCH = 4
    TAGS_NUM = 1

    def __init__(self, scope_name, iterator, embedding, batch_size):
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        with tf.variable_scope(scope_name) as scope:
            self._build_net()
    def _build_net(self):
        self.global_step = tf.Variable(0, trainable=False)






if __name__ =="__main__":

    """
    data = read_corpus('../zh-NER-TF/data_path/train_data')
    vocab_build('../zh-NER-TF/data_path/wore2id_new.pkl','../zh-NER-TF/data_path/train_data',10)
    print(sentence2id('我爱你'))
    """

    #训练词向量

    data = read_corpus('../zh-NER-TF/data_path/train_data')
    contents = []
    labels = []
    s = 0
    for sent_, tag_ in data:

        for word in sent_:
            if word.isdigit():
                sent_.remove(word)
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                sent_.remove(word)
            label_ = [tag2label[tag] for tag in tag_]

        contents.append(sent_)
        labels.append(label_)

    #print(contents[0],labels[0])
    #训练
    #train_word2vec(contents)

    mod = Word2Vec.load('../zh-NER-TF/data_path/word2v_model.pkl')
    #print(mod.most_similar('髦'))
    #print(mod['髦'].shape)
    #构建字典
    w2index, w2vec, conbined = create_dictionaries(mod,contents)
    print(len(conbined))

    #文本内容经过填充到最大长度100，那么label也需要对应填充到最大长度100

    labels = sequence.pad_sequences(labels, maxlen=100)
    print(len(labels))

    x_train,x_test,y_train,y_test = train_test_split(contents,labels,test_size=0.33, random_state=42)


