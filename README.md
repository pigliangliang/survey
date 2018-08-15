# 情感分析概述

该项目主是对自然语言处理领域中的情感分析技术的一个总结整理与概述。主要包括以下
内容

## 方法概述
根据分析载体的不同，情感分析会涉及很多主题，包括针对电影评论，商品评价，新闻和博客等的情感分析。在情感分析领域，文本识别可以划分为积极和消极两类，或者积极，消极和中性等，分析的主要方法有  

基于字典分析

        字典分析运用了有标记词汇组成的字典，使用词法分析器将输入文本转化为单词序列，将每一个新的单词  
	与字典中的词汇进行匹配。如果有一个积极的匹配，分数加到输入文本的总分中。例如：如果戏剧性在字典中是一个积极的匹配，那么文本总分会递增，若相反，会将总分进行减少。 文本的分类取决于文本的总得分。

基于机器学习的分析

        机器学习的方式具有较高的适用性和准确性。在情感分析中主要使用的方式是监督学习方法。可以分为三个步骤：数据收集，预处理，训练分类。在训练过程中，需要提供一个标记语料库作为训练数据。分类器使用乙烯类特征向量对目标数据进行分类。在机器学习技术中，决定分类器准确率的关键是合适的特征选择。通常来说，使用Ngram模型，tf-idf，word2vec等。算法模型可以使用传统的机器学习方法，比如逻辑回归，决策树，SVM，贝叶斯等，也可以是CNN、LSTM等深度学习算法。机器学习技术面临很多挑战：分类器的设计，训练数据的获取，对未知短语的识别等，相比词典分析方式，在字典大小呈指数增长的时候依然可以很好的工作。 
        
        流程：
        中文分词：
        1）这是相对于英文文本情感分析，中文独有的预处理
        2）常用方法：基于词典、基于规则、基于统计、基于字标注、基于人工智能。
        3）常用工具：哈工大—语言云、东北大学NiuTrans统计机器翻译系统、中科院张华平博士ICTCLAS、波森科技、结巴分词、Ansj分词，HanLP。
        提取特征：
        1）文本中拿什么作为特征。
        2）常用方法：根据词性（adj、adv、v）、单词进行组合（unigram、bigram）、位置。
        3）使用词的组合表示文本，两种方式：词出现与否、词出现的次数。
        特征选择
        1）选择哪些特征，如果把所有的特征都作为特征计算，那计算量非常大，高维稀疏矩阵。
        2）常用方法：tf-idf，互信息等
        3）常用工具：word2vector ，doc2vec
        训练方法
        1）机器学习算法：贝叶斯，svm，xgboost，lr
        2）神经网络，CNN，lstm
        评价方法
        召回率：反应被判定为正例的样本占全部正例样本的比例。
        准确率：反映了分类器统对整个样本的判定能力——能将正的判定为正，负的判定为负 。
        一般的，传统使用词袋模型+支持向量机方式，需要经历特征选择等阶段，它偏向于使用工程特征或者极性转移规则来提高 准确率。而作为新型情感分析技术的深度神经网络可以省去         特征工程这一环节,可以端到端训练而几乎不用人工参与。



## 文献综述

1、2017年的survey  

    论文主要是阐述在情感分析方面的基本方法的survey。  
     Published in: 2017 International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC)
     文档位置：document目录。

    论文主要内容：
       特征提取方式：论文是关于英文的情感分析，提取方式可以参考。  
       分类方法：词典和机器学习方式。  
       机器学习方式：传统机器学习和神经网络方式。
         
2、论文：Apply Word Vectors for Sentiment Analysis of APP Reviews

    The 2016 3rd International Conference on Systems and Informatics (ICSAI 2016)

3、论文：Chinese Micro-Blog Sentiment Analysis Based on Multi-Channels Convolutional Neural Networ

    一般的在机器学习方法中主要使用词向量的形式输入到神经网络或者使用机器学习方法进行训练，得到相应的分类结果，在该论文中，为了能够提取到文本的隐含信息将词向量，词的位置向量，词性向量进行多种方式的组合，论文中采用四中方式组合。输入CNN网络，包含输入层，卷积，池化，合并，隐藏，输出层。

    输入层：四种拼接向量
    V1 =w⊕tag⊕position
    V2=w⊕tag
    V3 =w⊕position
    V4 =tag⊕position
    
    卷积层：cj =relu(W·Vij:i+h-1 +b)

    池化层：C^j=(^c1 ,^c2 ,...,c^d)

    合并层：C=C1 ⊕C2 ⊕...⊕Cn 

    隐藏层：R = rel u ( W h C^ h + b h )  为了提升特征表达能力，进行非线性激励。
    
    输出层：softmax ：loss=-1∑ylny^+λ‖θ‖2 使用交叉熵函数调整优化模型参数。



   	  
## 工具总结

### 开源工具包

python机器学习库：  

    scikit-learn：机器学习库。 

nlp处理工具：  

    nltk：python 自然语言处理库。 

    Ansj是由孙健（ansjsun）开源的一个中文分词器，为ICTLAS的Java版本，也采用了Bigram + HMM分词模型  

    Jieba是由fxsjy大神开源的一款中文分词工具，一款属于工业界的分词工具——模型易用简单、代码清晰可读，推荐有志学习NLP或Python的读一下源码。与采用分词模型Bigram + HMM 的ICTCLAS 相类似，Jieba采用的是Unigram + HMM。Unigram假设每个词相互独立，则分词组合的联合概率： 

    LTP是哈工大开源的一套中文语言处理系统，涵盖了基本功能：分词、词性标注、命名实体识别、依存句法分析、语义角色标注、语义依存分析等。  

    CoreNLP是由斯坦福大学开源的一套Java NLP工具，提供诸如：词性标注（part-of-speech (POS) tagger）、命名实体识别（named entity recognizer (NER)）、情感分析（sentiment analysis）等功能。 

    gensim：开源的第三方python工具包，用于从原始的非结构化文本中，无监督的学习到文本隐层的主题向量表达。支持TF-IDF、LSA、LDA和word2vec在内的多种主题模型算法，支持流式训练，并提供了诸如相似度计算，信息检索等一系列常用任务的api接口。  

特征提取

    词袋模型
    tf-idf方式
    一般先使用sklearn库中提供的词袋函数，结合tf-idf处理，提升分类算法性能。

文本向量化方法

    词袋方法 ：首先构造词典，将文本中对应位置出现的词语标记为该词语在文本中的出现的频次。可以使用sklearn库中的方法词袋和tf-idf方法
    缺点：维度灾难，如果词典中有10000个单词那么文本就需要10000维向量表示；无法保留词序信息；存在语义鸿沟。
    
    CBOW模型：可以使用gensim库训练
    该模型使用文本的中间词作为目标词，通过上下文预测目标词出现的概率。

    skip-gram模型：使用gensim库训练
    目标词向量预测上下文的概率。


   
###  API 接口
