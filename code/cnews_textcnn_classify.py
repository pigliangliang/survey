#author_by zhuxiaoliang
#2018-08-27 上午11:16


"""

方式一：基于词语，根据gensim生成的词向量，然后转化生成词典，以及词典索引，词向量，本文方式
方式二：基于字符（单个字）可以先生成词典，使用Counter方式获取词频高的分词，已经词典生成分词文本的索引，直接输入CNN嵌入层，训练生成词向量。
"""

import jieba
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import numpy as np
from keras.preprocessing.text import Tokenizer#将分词文本转化为索引形式
from keras.utils import to_categorical



def read_file(filename):
    contents,labels = [],[]
    with open(filename,'r') as f:
        try:
            for line in f.readlines():
                label ,content = line.strip().split('\t')
                contents.append(content)
                labels.append(label)
        except Exception as e:
            print(e)

    return  contents,labels

def tokenizer(contents):
    text = [ jieba.lcut(text.strip()) for text in contents]

    return text

def word2vec_train(text):

    model = Word2Vec(size=100,min_count=10,window=5,sg=0)
    model.build_vocab(text)
    model.train(text,total_examples=len(text),epochs=1)

    model.save('../data/cnews/cnews_w2v_model.pkl')


def create_dictionary(model=None,text=None):
    if model is not None and text is not None:
        gensim_dic = Dictionary()
        gensim_dic.doc2bow(model.wv.vocab.keys(),allow_update=True)
        w2index = {v:k+1 for k,v in gensim_dic.items()}
        w2vec = {word :model[word] for word in w2index.keys()}

        def word2id(text):#将分词后文本转化为字典索引的形式，并补齐
            data=[]
            for te in text:
                word_2_id =[]
                try:
                    #不在词典中的分词索引为0
                    for word in te:
                        word_2_id.append(w2index[word])
                except :
                    word_2_id.append(0)
                data.append(word_2_id)
            return data
        text = word2id(text)
        text = sequence.pad_sequences(text,maxlen=100)

        return w2index,w2vec,text
    else:
        print('data is None')


def getdata(wordindex,wordvec,contents,labels):
    n_symbols = len(wordindex)+1
    embedding_weights = np.zeros((n_symbols,100),dtype='float64')

    for word,index in wordindex.items():
        embedding_weights[index,:] = wordvec[word]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    x_train,x_test,y_train,y_test = train_test_split(contents,labels,test_size=0.25,random_state=33)

    return (n_symbols,embedding_weights,x_train,x_test,y_train,y_test)


#构建CNN网络

import tensorflow as tf


class TextCNN(object):

    embedding_dim = 100  # 词向量维度
    seq_length = 100 # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    #vocab_size =   # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 4  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    def __init__(self,n_symbols,embeddingweigths):
        #self.config = config
        self.vocab_size = n_symbols
        self.embeddingweigths = embeddingweigths

        self.input_x = tf.placeholder(tf.int32,[None,self.seq_length],name='input_x')
        self.input_y= tf.placeholder(tf.int32,[None,self.num_classes],name='input_y')
        self.keep_prob = tf.placeholder(tf.float64,name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):

            #词向量
            embedding = tf.Variable(self.embeddingweigths)#加载训练好的词向量,

            #改为随机初始化
            #embedding = tf.get_variable('embedding',[5000,64])
            print(embedding.shape)
            embedding_inputs = tf.nn.embedding_lookup(embedding,self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.num_filters, self.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float64))





def train(config):
    n_symbols, embedding_weights, x_train, x_test, y_train, y_test = config[0],config[1],config[2],config[3],config[4],config[5]
    tcnn = TextCNN(n_symbols,embedding_weights)
    print('字典长度',n_symbols)
    from keras.utils import to_categorical
    # 转化为onehot编码
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #b保存模型
    saver = tf.train.Saver()
    def batch_iter(x, y, batch_size=64):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        #indices = np.random.permutation(np.arange(data_len))
        #x_shuffle = x[indices]
        #y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x[start_id:end_id], y[start_id:end_id]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #embedding_weights = tf.Variable(embedding_weights)
        #print(type(embedding_weights))
        #sess.run(embedding_weights)
        count = 0

        
        for epoch in range(tcnn.num_epochs):
            print('epoch:' +str(epoch+1))


            print(y_train.shape,x_train.shape)
            batch_train = batch_iter(x_train,y_train,batch_size=256)
            for x_batch ,y_batch in batch_train:

                #print(x_batch.shape,y_batch.shape)
                loss_train ,acc_train = sess.run([tcnn.loss,tcnn.acc],feed_dict={tcnn.input_x:x_batch,tcnn.input_y:y_batch,tcnn.keep_prob:0.5})
                print(loss_train,acc_train)

                count+=1
                sess.run(tcnn.optim, feed_dict={tcnn.input_x:x_batch,tcnn.input_y:y_batch,tcnn.keep_prob:0.5})

                #验证
                total_loss = 0.0
                total_acc = 0.0
                if count%50 ==0:
                    batch_test = batch_iter(x_test,y_test)
                    for x_,y_ in batch_test:
                        batch_len = len(x_)
                        #print(batch_len)
                        loss, acc = sess.run([tcnn.loss, tcnn.acc], feed_dict={tcnn.input_x:x_,tcnn.input_y:y_,tcnn.keep_prob:1.0})
                        total_loss += loss
                        total_acc += acc
                    print("total_loss:{},total_acc:{}".format((total_loss/batch_len),(total_acc/batch_len)))

                    if acc_train>0.95:
                        saver.save(sess=sess, save_path='.')






if __name__=="__main__":

    """
    训练词向量
    """
    test_contents ,test_lables = read_file('../data/cnews/cnews.test.txt')
    train_contents,train_label  = read_file('../data/cnews/cnews.train.txt')
    val_contents,val_label = read_file('../data/cnews/cnews.val.txt')

    contents = test_contents+train_contents+val_contents
    labels = test_lables+train_label+val_label

    contents = tokenizer(contents)
    #word2vec_train(contents)


    model = Word2Vec.load('../data/cnews/cnews_w2v_model.pkl')

    word2index,word2vec,contents = create_dictionary(model=model,text=contents)

    #print(word2index)

    config = getdata(word2index,word2vec,contents,labels)
    train(config)
