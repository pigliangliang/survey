#author_by zhuxiaoliang
#2018-08-20 下午3:05

import fasttext
'''


#train
#classfier=fasttext.supervised('news_fasttext_train.txt','news_fasttext.model',label_prefix='__label__')


#test

clf = fasttext.load_model('news_fasttext.model.bin',label_prefix = '__label__')
rel = clf.test('news_fasttext_test.txt')
print(rel.precision)
print(rel.recall)
'''

#测试
clf = fasttext.load_model('news_fasttext.model.bin')
text = ['最高人民法宣宣布周某某因涉嫌贪污受贿，利用不正当手段为他人谋取各种利益等，判处其无期徒刑，剥夺政治权利终身。',
        '婚姻大事不必铺张浪费',
        '小编祝大家新年快乐',
       '中国大陆多次强调，编排出武力夺取台湾' ,
        '它被誉为天下第一果，补益气血，养阴生津，现在吃正应季!  六七月是桃子大量上市的季节，因其色泽红润，肉质鲜美，有个在实验基地里接受治疗的妹子。广受大众的喜爱。’']
label = clf.predict(text)
print(label)
'''




#训练词向量

model = fasttext.skipgram('news_fasttext_train.txt','model1')
print(model.words)
'''