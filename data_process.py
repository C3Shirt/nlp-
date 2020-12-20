import numpy as np
import pandas as pd
import jieba


# define hyper-parametrs
DATA_PATH = './data/weibo_senti_100k.csv'
STOP_WORDS_PATH = './data/cn_stopwords.txt'


def data_n_fold(n_fold: int):
    """
    把需要的数据分为训练集和测试集，为了方便n折交叉验证，这里把数据分为n等份
    :param n_fold: data to n fold
    :return: 返回fold_data是一个list，包含十个元素，每一个元素是一个size为data_num/fold_num的字典
        这里的data_num是指文本的个数，fold_num是指对于全部数据分为fold_num份
        字典中包含两个k-v，key是'label'和'review'，value都是一个list，这两个list是一一对应
        的关系，其中review的value列表每一个值为一个评论文本，label的value列表每一个值为一个评论
        对应的label值
    """
    all_data = []
    data_df = pd.read_csv(DATA_PATH, sep=',')
    labels = data_df['label'].tolist()
    reviews = data_df['review'].tolist()

    patition_num = data_df[data_df['label'] == 1].shape[0] # 两种标签数据的分割线
    data_len = len(labels)
    all_index = np.arange(data_len)
    diff_index = []
    diff_index.append(all_index[:patition_num])
    diff_index.append(all_index[patition_num:])
    diff_n_fold = [[] for i in range(len(diff_index))]

    # 对于正面和负面的数据进行随机打乱，再分别进行n等分
    for i in range(len(diff_index)):
        np.random.shuffle(diff_index[i])
    for i in range(len(diff_index)):
        data_index = diff_index[i]
        num = len(data_index) // n_fold
        for j in range(n_fold):
            start_index = j * num
            end_index = num * (j + 1)
            if j != n_fold - 1:
                diff_n_fold[i].append(data_index[start_index:end_index])
            else:
                diff_n_fold[i].append(data_index[start_index:])

    # 把分别n等分的正面和负面数据进行合并并且随机打乱顺序
    fold_index = [[] for i in range(n_fold)]
    for i in range(n_fold):
        for j in range(len(diff_n_fold)):
            fold_index[i].extend(diff_n_fold[j][i])
        np.random.shuffle(fold_index[i])

    fold_data = []
    for index in range(n_fold):
        shuffle_fold_labels = [labels[i] for i in fold_index[index]]
        shuffle_fold_reviews = [reviews[i] for i in fold_index[index]]
        data = {'label': shuffle_fold_labels, 'review': shuffle_fold_reviews}
        fold_data.append(data)

    print("split data to %d fold" % n_fold)
    print("Fold lens %s" % str([len(data['label']) for data in fold_data]))

    return fold_data


def segmentation(fold_data: list):
    """
    把fold_data中的review进行分词并且去停用词
    :param fold_data: see return of data_n_fold function
    :return: None
    """
    stopwords_list = [line.strip() for line in open(STOP_WORDS_PATH, 'r').readlines()]

    for i in range(len(fold_data)):
        disposed_data = []
        data = fold_data[i]['review']
        for review in data:
            disposed_review = sent_dispose(review, stopwords_list)
            disposed_data.append(disposed_review)
        fold_data[i]['review'] = disposed_data


def sent_dispose(text: str, stopwords_list: list):
    """
    对于一个文本进行分词并且去停用词处理
    :param text: text need to segmentation and wipe off stop words
    :return: a list consists the disposed result
    """
    text_seg = jieba.lcut(text)
    disposed_text = []
    for word in text_seg:
        if word not in stopwords_list:
            if word is not " ":
                disposed_text.append(word)

    return disposed_text


def build_data(fold_data: list, dev_fold_index=None):
    """
    build the train data and dev data, we use the dev data as test data
    :param fold_data: see return of data_n_fold function
    :return: train_data and dev_data
             the type of train_data and dev_data is same: {'label': value-list, 'review': value-list}
    """
    if dev_fold_index is None:
        dev_fold_index = len(fold_data) - 1

    print("dev_fold_index: %d" % dev_fold_index)
    dev_data = fold_data[dev_fold_index]

    train_labels = []
    train_reviews = []
    train_fold_index = [i for i in range(dev_fold_index)]
    train_fold_index += [i for i in range(dev_fold_index + 1, len(fold_data))]
    print(train_fold_index)

    for fold_index in train_fold_index:
        train_labels.extend(fold_data[fold_index])
        train_reviews.extend(fold_data[fold_index])
    train_data = {'label': train_labels, 'review': train_reviews}

    return train_data, dev_data


def build_word2vec_data(fold_data:list):
    """
    为训练Word2Vec建立数据集
    :param fold_data: see return of data_n_fold function
    :return: a list include all reviews
    """
    train_reviews = []
    for i in range(len(fold_data)):
        train_reviews.extend(fold_data[i]['review'])
    print("Total %d reviews" % len(train_reviews))

    return train_reviews


if __name__ == "__main__":
    fold_data = data_n_fold(10)
    segmentation(fold_data)
    # print(fold_data[0]['review'])
    train_review = build_word2vec_data(fold_data)
    print(train_review)


