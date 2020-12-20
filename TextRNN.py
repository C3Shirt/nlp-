import numpy as np
import torch
import torch.nn as nn
import T2_SentimentAnalysis.data_process as DP
import time
import random
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score


WORD2VEC_PATH = './data/word2vec.txt'
SAVE_MODEL_PATH = './model.bin'
SAVE_TEST_PATH = './test_model.bin'


def set_cuda(gpu_id=0):
    use_cuda = gpu_id >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device("cpu")
    print("Use cuda: %s, gpu id: %d" % (use_cuda, gpu_id))
    return use_cuda, device


 def get_examples(data, vocab, max_sent_len=260):
        """
        :param data: 需要划分的全部评论数据
        :param vocab: 词表
        :param max_sent_len: 最大的句子长度
        :return: 一个list，每个元素是一个tuple(label, sent_len, extword_id)
        """

        examples = []
        for review, label in zip(data['review'], data['label']):
            id = int(label)
            # segments是一个list，其中每个元素是一个tuple(sent_len, sent_word)
            sent_len = len(review)
            sent_words = review
            extword_ids = vocab.extword2id(sent_words)
            examples.append((id, sent_len, extword_ids))

        return examples

    def batch_slice(data, batch_size):
        """
        build data loader
        把数据分割为多个batch，组成一个list返回
        :param data: 是get_examles()得到的examples
        :param batch_size: 批次大小
        :return:
        """

        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for i in range(batch_num):
            if i < (batch_num - 1):
                batch_review = data[(i * batch_size):((i + 1) * batch_size)]
            else:
                batch_review = data[(i * batch_size):]

            # 和return类似，也可以返回数据，但是是分批次的返回，可以节省内存
            # 用next)可以接着上次继续执行再返回数据
            yield batch_review

    def data_iter(data, batch_size, shuffle=True, noise=1.0):
        """
        在迭代训练中，调用data_iter函数，生成每个批次的batch_data，这个函数
        中会调用batch_slice来获取batch_data的原始数据

        :param data: get_examples()得到的结果，格式见get_example的返回
        :param batch_size: 批次大小
        :param shuffle: 是否为乱序
        :param noize:
        :return:
        """
        batched_data = []
        if shuffle:
            # 打乱所有数据
            np.random.shuffle(data)
            # lengths表示每篇文章的句子数量
            lengths = [example[1] for example in data]
            # 不知道这步有什么实质性的作用？
            noisy_lengths = [- (l + np.random.uniform(- noise, noise))
                             for l in lengths]
            sorted_indices = np.argsort(noisy_lengths).tolist()
            sorted_data = [data[i] for i in sorted_indices]
        else:
            sorted_data = data

        # 把batch的数据放在一个list中
        batched_data.extend(list(batch_slice(sorted_data, batch_size)))

        if shuffle:
            np.random.shuffle(batched_data)

        for batch_data in batched_data:
            yield batch_data


class Vocab():
    """
    通过训练好的word2vec文件建议词表
    """
    def __init__(self):
        self.pad = 0
        self.unk = 1
        self._id2extword = ['PAD', 'UNK']

    def load_pretrained_embs(self, word2vec_path):
        """
        加载预训练文件，并做一定的处理
        通过index在self._id2extword中可以找到word同时，通过index可以在embeddings中找到词向量
        同一个index找到的word和词向量是对应的对于不在词表中的word采用所有出现的词向量的平均值代替
        :param embfile_path: 预训练好的词向量文件路径
        :return: embeddings
        """
        with open(word2vec_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            items = lines[0].split()
            word_count, embed_dims = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((index + word_count, embed_dims), dtype = np.float32)
        for line in lines[1:]:
            value = line.split()
            self._id2extword.append(value[0])
            vector = np.array(value[1:], dtype=np.float64)
            embeddings[index] = vector
            embeddings[self.unk] += vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        embeddings = torch.from_numpy(embeddings)

        return embeddings

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    @property
    def extword_size(self):
        return len(self._extword2id)


class LSTMEncoder(nn.Module):
    """
       使用一个双层的LSTM对于单词进行编码，这里采取的是BiLSTM + Attention 对句子进行编码
       """
    def __init__(self, embeddings, dropout=0.15, hidden_size=128):
        super(LSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dims = 100

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.BiLSTM = nn.LSTM(input_size=self.embed_dims,
                              hidden_size=hidden_size,
                              num_layers=2,
                              bias=True,
                              batch_first=True,
                              dropout=0.15,
                              bidirectional=True)

    def forward(self, extword_ids, batch_masks):
        """
           输入一个批次的数据，其中包括extword_embed的indexs，batch_masks指出当前的这个word是否为padding word，
           如果是则相应位置为0，不是则为1
           :param extword_ids: (batch_size * sent_nums(max_doc_len), max_sent_len)
           :param batch_masks: 同上
           :return: word_hiddens
        """
        batch_embed = self.embedding(extword_ids)

        if self.training:
            batch_embed = self.dropout(batch_embed)

        output, _ = self.BiLSTM(batch_embed)
        word_hiddens = output * batch_masks.unsqueeze(2)

        if self.training:
            word_hiddens = self.dropout(word_hiddens)

        return word_hiddens


class Attention(nn.Module):
    def __init__(self, bihidden_size):
        super(Attention, self).__init__()
        self.query = nn.Parameter(torch.Tensor(bihidden_size))
        nn.init.normal_(self.query, mean=0.0, std=0.05)
        self.linear = nn.Linear(bihidden_size, bihidden_size, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch_hidden, batch_masks):

        key = self.linear(batch_hidden)

        attn_value = torch.matmul(key, self.query)

        # batch_masks: batch_size * len
        # 1 - batch_masks 就是取反，把没有单词的句子置为 0
        # masked_fill 的作用是在为1的地方替换为value: float(-1e32)
        mask_attn_value = attn_value.masked_fill((1 - batch_masks).bool(), float(-1e32))

        # attn_weights: batch_size * doc_len
        attn_weights = F.softmax(mask_attn_value, dim=1)

        # 其实这步就是把最后的填充句子的注意力权重置为0。其实在这里可以不做这个处理，因为经过之前的
        # mask_weights的处理之后，填充句子的部分注意力权重已经很小，接近于0了
        masked_attn_weights = attn_weights.masked_fill((1 - batch_masks).bool(), 0.0)

        # 为什么这里是对attn_middle进行求和而不是对原始的batch_hidden加权求和？
        # masked_attn_weights.unsqueeze(1): batch_size * 1 * doc_len
        # attn_middle: batch_size * doc_len * hidden(512)
        # batch_outputs: batch_size * hidden(512)
        reps = torch.bmm(masked_attn_weights.unsqueeze(1), batch_hidden).squeeze(1)

        return reps, attn_weights


class Model(nn.Module):
    """
    把SentEncoder、Attention、FC拼接起来搭建整体网络层
    """

    def __init__(self, vocab, use_cuda, device, extword_embed):
        super(Model, self).__init__()
        self.word_hidden_size = 128
        self.sent_reps_size =256
        self.all_parameters = {}
        self.dropout = 0.15

        parameters = []
        self.word_encoder = LSTMEncoder(extword_embed, self.dropout, self.word_hidden_size)
        self.word_attention = Attention(bihidden_size=self.sent_reps_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))

        # 最后加一层线性网络，利用doc_reps进行分类
        self.out = nn.Linear(self.sent_reps_size, 2, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        print('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        """
        :param batch_inputs: (batch_inputs1, batch_inputs2, batch_masks)
        :return: batch_outputs：batch_size * num_labels
                其实就是对于每个新闻文本预测出对于14个标签的概率分布
        """

        # batch_inputs : batch_size * sent_len
        # batch_masks : batch_size * sent_len
        batch_inputs, batch_masks = batch_inputs
        batch_size, max_sent_len = (batch_inputs.shape[0],
                                    batch_inputs.shape[1])
        batch_inputs = batch_inputs.view(batch_size, max_sent_len)
        batch_masks = batch_masks.view(batch_size, max_sent_len)

        # sent_reps: (sentence_num , sentence_rep_size)
        # (sen_num, <2 * lstm_hidden_size>) =  (sen_num , 256)
        word_hiddens = self.word_encoder(batch_inputs, batch_masks)
        sent_reps, word_atten_scores = self.word_attention(word_hiddens, batch_masks)

        # batch_size * num_labels
        batch_outputs = self.out(sent_reps)

        return batch_outputs


class Optimizer():
    """
       定义优化器类，对参数进行优化
       """

    def __init__(self, model_parameters, hyper_params):
        super(Optimizer, self).__init__()
        self.all_params = []
        self.optims = []
        self.schedulers = []
        self.lr = hyper_params['LR']
        self.decay = hyper_params['DECAY']
        self.decay_step = hyper_params['DECAY_STEP']

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=self.lr)
                self.optims.append(optim)

                l = lambda step: self.decay ** (step // self.decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim,
                                                              lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")
        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = '  %.5f' * self.num
        res = lr % lrs
        return res


class Trainer():
    def __init__(self):
        self.use_cuda, self.device = set_cuda(0)

        self.vocab = Vocab()
        self.extword_embed = self.vocab.load_pretrained_embs(WORD2VEC_PATH)
        self.model = Model(self.vocab, self.use_cuda, self.device, self.extword_embed)

        self.report = True
        self.save_model = SAVE_MODEL_PATH
        self.save_test = SAVE_TEST_PATH

        self.train_batch_size = 128
        self.test_batch_size = 128

        fold_data = DP.data_n_fold(10)
        DP.segmentation(fold_data)
        train_data, dev_data = DP.build_data(fold_data)
        self.train_examples = get_examples(train_data, self.vocab)
        self.dev_examples = get_examples(dev_data, self.vocab)
        # 把数据分成batch进行训练，每个批次大小为batch_size，这里的batch_num为一个有多少个批次
        self.batch_num = int(np.ceil(len(self.train_data) /
                                     float(self.train_batch_size)))

        # count
        self.epochs = 20
        self.early_stops = 3
        self.log_interval = 50
        self.clip = 5.0
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = self.epochs

        # define criterion
        self.criterion = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = Optimizer(self.model.all_parameters, hyper_params)

        print("train init finish")

    def train(self):
        print('Start trainning...')
        for epoch in range(1, self.epochs + 1):
            train_f1 = self._train(epoch)
            dev_f1 = self._eval(epoch)

            if self.best_train_f1 <= dev_f1:
                print("Exceed history dev = %.2f, current dev = %.2f" %
                      (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), self.save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.early_stops:
                    print("Early stop in epoch %d, best train: %.2f, best dev: %.2f" %
                          (epoch, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()   # see source code 其实就是把model以及子模型的trainning = True
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []

        for batch_data in data_iter(self.train_examples, self.train_batch_size,
                                                      shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            # 把预测值转换为一维，方便之后计算f1-score
            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            # 为了防止梯度爆炸，这里采用梯度裁剪
            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()

            self.optimizer.zero_grad()
            self.step += 1

            if batch_idx % self.log_interval == 0:
                lrs = self.optimizer.get_lr()
                print(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f}}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / self.log_interval))

                losses = 0

            batch_idx += 1

        overall_losses = overall_losses / self.batch_num

        # 保留4位小数
        overall_losses = float(format(overall_losses, '.4f'))
        f1 = self.get_score(y_pred, y_true)

        print(
            '| epoch {:3d} | f1 {} | loss {:.4f}}'.format(epoch, f1, overall_losses))

        return f1

    def _eval(self, epoch, test=False):
        self.model.eval()   # see source code 其实就是把model以及子模型的trainning = False
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in DataProcessLoader.data_iter(data, self.test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            f1 = self.get_score(y_pred, y_true)

            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(self.save_test, index=False, sep=',')
            else:
                print('| epoch {:3d} | dev | f1 {} | time {:.2f}'.format(epoch, f1, during_time))

        return f1

    def batch2tensor(self, batch_data):
        batch_size = len(batch_data)
        sent_labels = []
        sent_lens = []
        sent_len_list = []
        for sent_data in batch_data:
            sent_labels.append(sent_data[0])
            sent_lens.append(sent_data[1])
            # 取出这篇新闻中最长的句子长度，添加到列表中
            sent_len_list.append(max(sent_lens))

        # 取出这批最长句子长度
        max_sent_len = max(sent_len_list)

        # 创建用于训练的数据的格式
        batch_inputs = torch.zeros((batch_size, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(sent_labels)

        for b in range(batch_size):
                sent_data = batch_data[b][2]
                for word_idx in range(batch_data[b][1]):
                    batch_inputs[b, word_idx] = sent_data[word_idx]
                    batch_masks[b, word_idx] = 1

        if self.use_cuda:
            batch_inputs = batch_inputs.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_labels = batch_labels.to(self.device)

        return (batch_inputs, batch_masks), batch_labels

    def get_score(self, y_pred, y_true):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        return f1


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()