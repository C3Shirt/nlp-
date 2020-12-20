import numpy as np
from gensim.models.word2vec import Word2Vec
import T2_SentimentAnalysis.data_process as DP

# define hyper-parameters
NUM_FEATURES = 100 # the dimension of word vector
NUM_WORKER = 8  # number of thread to run parallelly
SAVE_FILE = './data/word2vec.txt'


def train_word2vec():
    """
    train the word2vec model and save the result
    :return: word2vec result
    """
    fold_data = DP.data_n_fold(10)
    DP.segmentation(fold_data)
    train_reviews = DP.build_word2vec_data(fold_data)
    model = Word2Vec(train_reviews, size=NUM_FEATURES, workers=NUM_WORKER, min_count=2)
    model.wv.init_sims(replace=True)

    model.wv.save_word2vec_format(SAVE_FILE, binary=False)


if __name__ == "__main__":
    train_word2vec()
