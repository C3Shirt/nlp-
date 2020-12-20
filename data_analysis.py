import pandas as pd
import matplotlib.pyplot as plt

"""
对数据集进行数据分析，只有了解了数据的分布才能更好的设计模型对数据进行拟合
"""

data_set = pd.read_csv('./data/weibo_senti_100k.csv', sep=',')

print(data_set.shape)
print(data_set.sample(10))
print(data_set.info())

print("label 0 count:", data_set[data_set['label'] == 0].shape[0])
print("label 1 count:", data_set[data_set['label'] == 1].shape[0])
# 正面和负面情感数据数量基本一致

# 文本的长度分布
data_set['review_length'] = data_set['review'].apply(lambda x: len(x))
print("max length of positive review: ", data_set[data_set['label'] == 1]['review_length'].max())
print("min length of positive review: ", data_set[data_set['label'] == 1]['review_length'].min())
print("min length of negative review: ", data_set[data_set['label'] == 0]['review_length'].max())
print("min length of negative review: ", data_set[data_set['label'] == 0]['review_length'].min())
print("length description: ", data_set['review_length'].describe())

# plt.hist(data_set['review_length'], bins=50)
# plt.title('distribution of review length')
# plt.show()
