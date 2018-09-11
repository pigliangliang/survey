#author_by zhuxiaoliang
#2018-08-30 下午7:14

import jieba

def get_txt_data(filepath, para):
    if para == 'lines':
        txt_file1 = open(filepath, 'r')
        txt_tmp1 = txt_file1.readlines()
        txt_tmp2 = ''.join(txt_tmp1)
        txt_data1 = txt_tmp2.split('\n')
        txt_file1.close()
        return txt_data1
    elif para == 'line':
        txt_file2 = open(filepath, 'r')
        txt_tmp = txt_file2.readline()
        txt_data2 =' '.join(txt_tmp)
        txt_file2.close()
        return txt_data2
#切缝句子
def cut_sentence_2(words):
    start = 0
    i = 0 #i is the position of words
    token = 'meaningless'
    sents = []
    punt_list = ',.!?:;~，。！？：；～…  '
    for word in words:
        if word not in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
            #print token
        elif word in punt_list and token in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
        else:
            sents.append(words[start:i+1])
            start = i+1
            i += 1
    if start < len(words):
        sents.append(words[start:])
    return sents



"""
Compute a review's positive and negative score, their average score and standard deviation.
This module aim to extract review positive/negative score, average score and standard deviation features (all 6 features).
Sentiment analysis based on sentiment dictionary.

"""
import logging
import numpy as np


# 1. Load dictionary and dataset
# Load sentiment dictionary
#negdict = get_txt_data("../sentiment dictionary/positive and negative dictionary//negdict.txt", "lines")
my_posdict = get_txt_data("../sentiment dictionary/positive and negative dictionary/my_posdict.txt", "lines")
my_negdict = get_txt_data("../sentiment dictionary/positive and negative dictionary/my_negdict.txt", "lines")
# Load adverbs of degree dictionary
mostdict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/most.txt', 'lines')
verydict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/very.txt', 'lines')
moredict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/more.txt', 'lines')
ishdict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/ish.txt', 'lines')
insufficientdict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/insufficiently.txt', 'lines')
inversedict = get_txt_data('../sentiment dictionary/adverbs of degree dictionary/inverse.txt', 'lines')


# Load dataset
# review = tp.get_excel_data("../Machine learning features/seniment review set/pos_review.xlsx", 1, 1, "data")


# 2. Sentiment dictionary analysis basic function
# Function of matching adverbs of degree and set weights
def match(word, sentiment_value):
    if word in mostdict:
        logging.info('most word: %s' % word)
        sentiment_value *= 2.0
    elif word in verydict:
        logging.info('very word: %s' % word)
        sentiment_value *= 1.5
    elif word in moredict:
        logging.info('more word: %s' % word)
        sentiment_value *= 1.25
    elif word in ishdict:
        logging.info('ish word: %s' % word)
        sentiment_value *= 0.5
    elif word in insufficientdict:
        logging.info('insufficient word: %s' % word)
        sentiment_value *= 0.25
    elif word in inversedict:
        logging.info('inverse word: %s' % word)
        sentiment_value *= -1
    return sentiment_value


# Function of transforming negative score to positive score
# Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
def transform_to_positive_num(poscount, negcount):
    pos_count = 0
    neg_count = 0
    if poscount < 0 and negcount >= 0:
        neg_count += negcount - poscount
        pos_count = 0
    elif negcount < 0 and poscount >= 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount < 0:
        neg_count = -poscount
        pos_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
    return [pos_count, neg_count]


# 3.1 Single review's positive and negative score
# Function of calculating review's every sentence sentiment score
def sumup_sentence_sentiment_score(score_list):
    score_array = np.array(score_list)  # Change list to a numpy array
    Pos = np.sum(score_array[:, 0])  # Compute positive score
    Neg = np.sum(score_array[:, 1])
    AvgPos = np.mean(score_array[:, 0])  # Compute review positive average score, average score = score/sentence number
    AvgNeg = np.mean(score_array[:, 1])
    #StdPos = np.std(score_array[:, 0])  # Compute review positive standard deviation score
    #StdNeg = np.std(score_array[:, 1])

    return [Pos, Neg, AvgPos, AvgNeg]


def single_review_sentiment_score(review):
    single_review_senti_score = []
    cuted_review = cut_sentence_2(review)
    for sent in cuted_review:
        seg = jieba.cut(''.join(sent))

        seg_sent=[]
        for s in seg:
            seg_sent.append(s)
        print("------",seg_sent,'-----------')
        i = 0  # word position counter
        s = 0  # sentiment word position
        poscount = 0  # count a positive word
        negcount = 0  # count a negative word
        neg_times = 0
        for word in seg_sent:
            # 排除一些空字符
            if len(word.strip()) == 0:
                continue
            logging.info('word : %s' % word)
            poscount_temp = 0
            negcount_temp = 0
            if word in my_posdict:
                #print(word)
                logging.info("pos word : %s" % word)
                poscount_temp += 1
                for w in seg_sent[s:i]:
                    # 排除一些空字符
                    if len(w.strip()) == 0:
                        continue
                    poscount_temp = match(w, poscount_temp)
                poscount += poscount_temp
            elif word in my_negdict:

                #print(word)
                logging.info("neg word : %s" % word)
                negcount_temp += 1
                # neg_times += 1
                for w in seg_sent[s:i]:
                    # 排除一些空字符
                    if len(w.strip()) == 0:
                        continue
                    negcount_temp = match(w, negcount_temp)
                negcount += negcount_temp
            # Match "!" in the review, every "!" has a weight of +2
            elif word == "！"or word == "!":
                for w2 in seg_sent[::-1]:
                    if w2 in my_posdict:
                        poscount += 2
                        break
                    elif w2 in my_negdict:
                        negcount += 2
                        break
            elif word == u"？" or word == u"?":
                poscount = 0
                negcount = 0
            i += 1


        single_review_senti_score.append(transform_to_positive_num(poscount, negcount))
    # pdb.set_trace()
    review_sentiment_score = sumup_sentence_sentiment_score(single_review_senti_score)

    return review_sentiment_score


import pandas as pd

def read_file(path=None):

    csv_data = pd.read_csv('../data/news_crawl_temp.csv',header=None,index_col=None)

    return  csv_data


# Testing
# for i in all_review_sentiment_score(sentence_sentiment_score(review)):
# 	print i


# 4. Store sentiment dictionary features


if __name__ =="__main__":
    """
    #s = cut_sentence_2("经济观察网记者李紫宸 8月29日上午，中华全国工商业联合会在辽宁沈阳发布2018中国民营企业500强榜单。过去一年中（2017年），民营企业500强入围门槛为156.84亿元，500强营收总额为244793.82亿元，此前一年这一数字为193616.14亿元。其中，营收总额超过1000亿元的有42家，上一年为27家，500到1000亿之间的则有91家，上一年为64家。2017年，民营企业500强产业结构中第二产业仍占主体地位，但入围企业数量有所减少。从入围企业数量来看，民营企业500强仍以制造业为主导，钢铁行业仍居前列，但数量较上一年度有所增加，综合业首次成为第二大主体，民营企业500强行业结构进一步优化。从资产规模来看，民营企业500强产业结构延续往年态势，第二产业资产规模占比继续降低，第三产业资产规模占比持续上升。各行业经营效益整体有所提升，煤炭开采和洗选业，医药制造业，铁路、船舶、航空航天和房地产业的销售净利率、资产净利率和净资产收益率相对较高；货币金融服务业，畜牧业等行业经营效益较上年明显下降。具体来看，本年度民营企业500强企业呈现以下主要特点：一、服务业比重增强，产业结构优化升级第一产业入围企业数量为5家，与上一年持平。第二产业入围企业数量连续五年下降，从2012年的380家降至2017年的333家，下降幅度为14.11%。第三产业入围企业数量则连续五年增加，从2012年的117家增至2017年的162家6%。第二产业资产总额113844.86亿元，占40.38%，同比减少0.02个百分点，民营企业500强中第二产业企业资产比逐年递减。二、制造业仍占主导地位2017年，民营企业500强中制造业仍占主导地位，民营企业500强有88家制造业企业，较上一年持平。制造业企业营业收入占民营企业500强比重同比下降0.31个百分点，资产总额占比同比增长0.15个百分点，缴税额占比同比下降1.06个百分点，税后净利润占比同比增长2.58个百分点，从业人员占比同比下降2.79个百分点，研发费用占比同比下降4.3个百分点。调研显示，在经济下行压力加大的背景下，民营企业500强中的制造业企业研发投入仍保持较高占比，原始创新能力不断提升。三、前十大行业结构出现变化2017年，民营企业500强前十大行业共包含11个行业316家企业，较上一年减少2个行业22家。行业分布继续变化，黑色金属冶炼和压延加工业跃居第一，综合排名提升到第二位，建筑业排名下降到第三位，房地产业位居第四位。建筑业，房地产业，批发业及有色金属冶炼和压延加工业入围企业数量均出现不同程度的减少。黑色金属冶炼和压延加工业，综合，化学料和化学制品制造业，电气机械和器材制造业，石油加工、炼焦和核燃料加工业与零售业等行业入围企业数量则有明显增加，前十大行业结构变化反映民营企业500强行业集中度进一步提高，呈现传统产业向新兴产业调整的趋势。2018中国民营企业500强榜单、2018中国民营企业制造业500强榜单、2018中国民营企业服务业100强榜单（部分）如下：")
    s = '8月25号上午，22岁的女大学生小丁从南通市区回海安老家，因为赶时间，她乘坐了一辆广东牌照的滴滴顺风车，一路上不是没有遭到司机人身攻击和辱骂。顺风车司机说没有对乘客殴打'
    s = '8月31日，今日两市低开后一路震荡下行，沪指一度跌近1%，再次考验2700点整数关口，创业板指跌幅逾1.5%，月线上四连阴，盘中在银行等金融股强力护盘下，三大股指虽然悉数反弹，且沪指率先翻红，但午后各大股指再度疲软，持续走低，全天维持宽幅震荡走势，本周沪指累计跌幅0.15%。从盘面上看，知识产权保护、银行居等板块位居涨幅榜前列，在线旅游、采掘服务、次新股等板块跌幅居前。'
    Pos, Neg, AvgPos, AvgNeg,   = single_review_sentiment_score(s)
    print(Pos,Neg)
    print("积极分{}，消极分{}".format(Pos,Neg))
    """
    #对news读取内容
    contents = read_file()
    score = []
    for content in contents[0]:
        #print(content)
        Pos, Neg,_,_ = single_review_sentiment_score(content)
       # print(Pos, Neg)
        print("积极分{}，消极分{}".format(Pos, Neg))
        #score 写入文件

        if Pos>Neg :
            score.append(1)
        elif Pos<Neg :
            score.append(-1)
        else:
            score.append(0)
    with open('score.txt','w') as f:
        for i in score:
            f.write(str(i))
            f.write('\n')


