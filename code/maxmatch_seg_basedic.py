class IMM(object):
    def __init__(self,dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(dic_path,'r',) as f:
            for x in (f.read().split()):
                self.dictionary.add(x)
                if len(x)>self.maximum:
                    self.maximum = len(x)
    def cut(self,text):
        result = []
        index = len(text)
        print(index)
        while index:
            word = None
            if index-self.maximum<=0:
                piece=text[:index]
                if piece in self.dictionary:
                    word = piece
                    result.append(text[:index])
                    text = text[index:]
                    index = len(text)
                if index==1:
                    text=text[index:]
                    index=len(text)
            if word is None:
                index -=1
        return result
if __name__ == '__main__':
    text = '我在杭州西湖边的西湖博物馆里面。'
    tokenizer = IMM('imm_dic')
    print(tokenizer.dictionary)
    print(tokenizer.maximum)
    print(tokenizer.cut(text))

