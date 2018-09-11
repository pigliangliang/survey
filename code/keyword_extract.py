#author_by zhuxiaoliang
#2018-08-20 下午4:54


#关键词提取
from jieba.analyse import textrank,tfidf

text = '之后不久,亲友联系这名姑娘时,发现她的手机已经处于关机状态。记者从乐清警方处确认了该起案件,滴滴司机已经落网,案件目前还在侦办中。警方通报 8月24日17时35分,乐清警方接群众报警称其女儿赵某(20岁、乐清人)于当日13时,在虹桥镇乘坐滴滴顺风车前往永嘉。14时许,赵某向朋友发送“救命”讯息后失联。接报后,乐清警方高度重视,立即启动重大案事件处置预案,全警种作战,并在上级公安机关的全力支持下,于25日凌晨4时许,在柳市镇抓获犯罪嫌疑人钟某(男、27岁、四川人)。经初步侦查,该滴滴司机钟某交代了对赵某实施强奸,并将其杀害的犯罪事实,目前案件正在进一步侦查中。'
tr = textrank
keywords = tr(text)
print('/'.join(keywords))

tf= tfidf
keywords2 = tf(text,topK=5)
print('/'.join(keywords2))







#主题模型

import  math

import jieba
import jieba.posseg as psg

from  gensim import models,corpora

text = '美团顺风车出现事故了，乐清女孩杀死'
seg = psg.cut(text)
st  = ''
for s in seg:
    if s.flag.startswith('n') or s.flag.startswith('a') or s.flag.startswith('v') or s.flag.startswith('d'):
        #if len(s.word)!=1:
           # print(s.word,end=' ')
            st +=s.word
print('888',st)
#print('#'*30)


#seg2 = jieba.cut(text)
#for s in seg2:

 #   print(s,end=' ')



def get_stopword_list():
    path = '../data/cnews/stopwords.txt'
    stopwords = [sw for sw in open(path,'r').readlines()]
    return  stopwords


