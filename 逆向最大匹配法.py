import re
class IMM(object):
    def __init__(self,dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(dic_path, 'r' ,encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.maximum = max(len(line),self.maximum)
        #这个初始化主要是读取字典的内容，同时记录最长词的长度
    def cut(self,text):
        result = []
        index = len(text) #这个获取的是最后一个汉字的索引
        while index > 0:
            word = None
            for size in range(self.maximum,0,-1):#然后依次用词库中最长的词去匹配
                if index - size <0: #索引index-size小于0，说明待匹配的部分没那么长了，就找短一点的词来匹配
                    continue
                piece = text[(index-size):index] #把待匹配的文本切片，切成和字典中的词一样的长度
                if piece in self.dictionary: #如果piece在字典中，那么就添加到结果中，同时移动index
                    word = piece
                    result.append(word)
                    index-=size
                    break
            if word is None: #word是None说明切出来的词不在字典里面，移动index
                index-=1
        return result[::-1]
def main():
    text = '南京市长江大桥'
    tokenizer = IMM('I:\py_NLP\imm_dic.utf8')
    print(tokenizer.cut(text))
main()
'''
南京市
南京市长
长江大桥
人名解放军
大桥
'''
