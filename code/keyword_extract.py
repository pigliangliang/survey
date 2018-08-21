#author_by zhuxiaoliang
#2018-08-20 下午4:54
'''

#关键词提取
from jieba.analyse import textrank,tfidf

text = '据半岛电视台援引叙利亚国家电视台称，叙利亚已经对美国、英国、法国的空袭进行了反击。据介绍，在叙军武器库中，对西方最具威慑力的当属各型战术地对地弹道导弹。尽管美英法是利用巡航导弹等武器发动远程空袭，但叙军要对等还击却几乎是“不可能完成的任务”。目前叙军仍能作战的战机仍是老旧的苏制米格-29、米格-23、米格-21战斗机和苏-22、苏-24轰炸机，它们在现代化的西方空军面前难有自保之力，因此叙军的远程反击只能依靠另一个撒手锏——地对地战术弹道导弹.'

tr = textrank
keywords = tr(text,topK=8)
print('/'.join(keywords))

tf= tfidf
keywords2 = tf(text,topK=8)
print('/'.join(keywords2))

'''




#NLP主题建模

from gensim.corpora import Dictionary
from gensim import models,similarities


with open('postrain.txt','r') as f:
    con= f.readlines()
content = [[word for word in line.split() ]for line in con]
#print(content)
#统计文档词典
dictionary = Dictionary(content)
print(dictionary)
#统计文档词频矩阵
te = [dictionary.doc2bow(text) for text in content]
print(len(te))
#print(te)#输出的是稀疏矩阵


#词袋处理后的结果，使用TFIDF算法处理后，可以进一步提升LDA的效果
tfidf = models.TfidfModel(te)[te]#转化为tfidf模型和向量形式
lda = models.LdaModel(corpus=tfidf,id2word=dictionary,num_topics=50)
lsi = models.LsiModel(corpus=tfidf,id2word=dictionary,num_topics=50)
#判断文本匹配的主题
for index,topic in lsi.print_topics(5):
    print (index,topic)
corpus_lsi = lsi[tfidf]
#print(corpus_lsi[1])
#排序
from collections import defaultdict
d = defaultdict(float)
for k,v in corpus_lsi[1]:
    d[k]=v
print(sorted(d.items(),key=lambda x:x[1],reverse=True))
print('*'*20)
for index,topic in lda.print_topics(num_topics=5):
    print (index,topic)




'''
这部分代码有问题，预期实现的功能是实现文档相似度分析
#参考文档：http://www.52nlp.cn/如何计算两个文档的相似度二
#给定text，测试其匹配的主题
text = '手机 不错 性价比 高'
text_bow = [dictionary.doc2bow([t for t in text ])]
#print(text_bow)
tfidf = models.TfidfModel(te)#转化为tfidf模型和向量形式
#使用训练好的lsi模型将其映射到二维的topic空间
#text_tfidf = lsi[tfidf]
#计算其和index中doc的余弦相似度了
index = similarities.MatrixSimilarity(tfidf[te])
sims = index[tfidf[text_bow]]
print(sims)

'''


#查询某个文档属于哪个主题,对比lsi模型

corpus_lda = lda[tfidf]
#print(len(corpus_lda))
print(corpus_lda[1])




#topic_list = lda.print_topic(4)

#print(type(topic_list))
#print(len(topic_list))

#print(lda.print_topic(1))
#for index,topic in lda.print_topics():
#    print (index,topic)
'''
corpus_lda = lda[tfidf]
print(corpus_lda[1])




for index,topic in lda.print_topics():
    print (topic)
print('8'*20)
for index,topic in lsi.print_topics(5):
    print (topic)
'''