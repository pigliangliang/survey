#author_by zhuxiaoliang
#2018-08-23 下午7:15

"""
分词
"""
import jieba
import os

stopwords_path =  '../data/cnews/stopwords.txt'

class WordCut:
    def __init__(self,stopwords_path=stopwords_path):

        self.stopwords_path = stopwords_path

    def addDictionary(self,dict_list):
        """
        添加用户自定义字典
        :param dict_list:
        :return:
        """
        map(lambda x:jieba.load_userdict(x),dict_list)

    def seg_sentence(self,sentence,stopwords_path=None):
        """
        句子分词
        :param sentence:
        :param stopwords_path:
        :return:
        """
        if stopwords_path is None:
            stopwords_path = self.stopwords_path
        stopwords = [line.strip() for line in open(stopwords_path,'r').readlines()]
        sentence_seg = jieba.lcut(sentence.strip())


        for word in sentence_seg:
            if word not in stopwords:
                continue
            else:
                sentence_seg.remove(word)
        return sentence_seg
    def seg_file(self,path):
        """
        对文件中文本分词
        :param path:
        :return:
        """
        file_seg = []
        with open(path,'r') as f:
            for line in f.readlines():
                line_seg = self.seg_sentence(line)
                file_seg.append(line_seg)

        return file_seg

if __name__ == "__main__":
    text = '黔清宫真是像大婶说的一样，一直到文昌市，一路都没看到卖椰子的。到文昌市区，已经是下午3点，我们早已饥肠辘辘，赶忙下车寻觅午餐。到了文昌当然要吃最有名的文昌鸡，而从当地人口中得知最好吃的做法就是盐焗鸡，市区里非常好找，沿路有不少的店面。'
    file_path = '../data/cnews/cnews.val.txt'
    wc = WordCut()
    print(wc.seg_sentence(text))
    file_contents = wc.seg_file(path=file_path)
    print(file_contents[:4])