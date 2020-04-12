import re
class MM(object):
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
    def cut(self,text):
        result = []
        index = 0 #这个获取的是最后一个汉字的索引
        length = len(text)
        while index < length: #限制条件
            word = None
            for size in range(self.maximum,0,-1):#然后依次用词库中最长的词去匹配
                if length - index+1 < size: #如果剩下的字符长度小于size，选择合适的size进行切分
                    continue
                piece = text[index:size+index] #切片从index开始取size大小的匹配字段
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                index += 1
        return result[::-1]
def main():
    text = '南京市长江大桥'
    tokenizer = MM('I:\py_NLP\imm_dic.utf8')
    print(tokenizer.cut(text))
main()
