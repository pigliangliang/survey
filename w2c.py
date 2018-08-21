#author_by zhuxiaoliang
#2018-08-19 下午6:02

from  gensim.models import Word2Vec
from gensim.models import word2vec



#两种形式的文本输入都是可以的
#sentences = ['我 爱你','你  很好']
sentences = [['first', 'sentence'], ['second', 'sentence']]
model = Word2Vec(sentences,min_count=1)
print(model['first'])





'''
sentences = word2vec.Text8Corpus('train.txt')

#model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)

model = word2vec.Word2Vec(sentences,size=50)

#print(model.similarity('差评','质量'))
#print(model.most_similar(u'不好'))
import numpy as np
z = []
z.append(model['质量'].tolist())
z.append(model['差'].tolist())
print(z[0])
#x = model['不错'].reshape(10,-1)
#print(x)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
y = ['1','-1']
#print(y)

lr.fit(z,y)
print(lr.score(z,y))#
print(lr.predict([z[0]]))#'''