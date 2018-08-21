#author_by zhuxiaoliang
#2018-08-21 下午2:28

from gensim import corpora,models,similarities




def load_content(path):
    with open(path,'r') as f:
        content = f.readlines()
    return content


def gensim_sim(content,test_text):

    x = [[word for word in line.split()] for line in content]
    #获取词袋
    dictionary = corpora.Dictionary(x)
    #制作预料
    corpus = [dictionary.doc2bow(doc) for doc in x]
    #tfidf处理
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    #把测试文旦转换为tfidf
    test_text_vec = [word for word in test_text.split()]
    test_text_vec = tfidf[dictionary.doc2bow(test_text_vec)]
    #判断和文本中相似的内容
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=len(dictionary.keys()))
    sim = index[tfidf[test_text_vec]]
    #print(sim)
    for index, score in sorted(enumerate(sim), key=lambda item: -item[1])[:6]:
        # print   "index:%d similarities:%f" % (index, score)
        print("index:%d similarities:%f content:%s" % (index, score, content[index]))
    '''
    输出：排在第一位的是和原文
    index:275 similarities:0.986117 content:值得 拥有

    index:454 similarities:0.906838 content:正品 值得 拥有

    index:316 similarities:0.616079 content:好用 超值 不错 质量 值得 拥有

    index:90 similarities:0.595103 content:值得 拥有 裸机 没 送
    
    index:269 similarities:0.575879 content:质量 杠杠 滴 值得 拥有
    
    index:515 similarities:0.435243 content:实在 慢 不行 速度 上网 值得 拥有
    
    '''
    #lsi
    lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=5)
    #output 5 topic
    for l in lsi.print_topics():
        print(lsi.print_topics())
    corpus_lsi = lsi[corpus_tfidf]
    #输出每个文档对应的主题概率分布
    #for doc in corpus_lsi:
     #   print(doc)
    print(corpus_lsi[0])
    print('-'*50)
    #lda
    lda = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=5)
    #输出5个主题
    for l in lda.print_topics():
        print(lda.print_topics())
    corpus_lda = lda[corpus_tfidf]
    #输出第一个文档对应的主题概率分布
    print(corpus_lda[0])

if __name__ == "__main__":
    path = '../postrain.txt'
    content = load_content(path)
    print(content[1])
    test = content[1]
    gensim_sim(content,test_text=test)