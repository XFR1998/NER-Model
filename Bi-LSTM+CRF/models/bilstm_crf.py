import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, configs):
        super(BiLSTM_CRF, self).__init__()
        self.configs = configs
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(configs.embedding_pretrained,
        #                                                   freeze=False)  # 表示训练过程词嵌入向量会更新
        # else:

        self.embedding = nn.Embedding(configs.vocab_len, configs.embedding_dim,
                                      padding_idx=configs.word2idx['<PAD>'])  # PAD索引填充

        if configs.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1


        self.rnn = nn.LSTM(input_size=configs.embedding_dim,
                           hidden_size=configs.hidden_size,
                           num_layers=configs.num_layers,
                           batch_first=True,
                           bidirectional=configs.bidirectional)

        self.tag2idx = configs.tag2idx

        # 转换参数矩阵 输入i,j是得分从j转换到i
        self.tagset_size = len(self.tag2idx)
        # 将lstm的输出映射到标记空间
        self.hidden2tag = nn.Linear(configs.hidden_size*self.num_directions, self.tagset_size)  # -> (B, num_class+2)  加上了START END
        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)

    def _init_hidden(self, batchs):  # 初始化h_0和c_0 与GRU不同的是多了c_0（细胞状态）
        h_0 = torch.randn(self.configs.num_layers*self.num_directions, batchs,  self.configs.hidden_size)
        c_0 = torch.randn(self.configs.num_layers*self.num_directions, batchs, self.configs.hidden_size)
        return self._make_tensor(h_0), self._make_tensor(c_0)

    def _get_lstm_features(self, x):
        # x.shape: (bs, num_words)
        x = self.embedding(x)
        # x.shape: (bs, num_words, embedding_dim)
        h_0, c_0 = self._init_hidden(batchs=x.size(0))
        out, (hidden, c) = self.rnn(x,(h_0, c_0))
        # out.shape: (bs, num_words, hidden_size*2)
        out = self.hidden2tag(out)  # (B,num_directions*hidden_size) -> (B, num_class)
        # out.shape: (bs, num_words, tagset_size)
        return out

    def neg_log_likelihood(self, sentence_tensor=None, label_tensor=None, mask_tensor=None):  # 损失函数
        feats = self._get_lstm_features(sentence_tensor)
        return -self.crf(emissions=feats, tags=label_tensor, mask=mask_tensor)
        # return -self.crf(emissions=feats, tags=label_tensor)


    def _make_tensor(self, tensor):
        # 函数说明： 将传入的tensor转移到cpu或gpu内
        tensor_ret = tensor.to(self.configs.device)
        return tensor_ret


    def forward(self, sentence_tensor=None, mask_tensor=None):
        # 数据预处理时，x被处理成是一个tuple,其内容是: (word, label).
        # x:b_size
        # print(sentence_tensor)
        # print(mask_tensor)
        lstm_feats = self._get_lstm_features(sentence_tensor)  # 获取BiLSTM的emission分数

        # Returns: List of list containing the best tag sequence for each batch.
        # 返回列表组成的标签

        out = self.crf.decode(emissions=lstm_feats,
                              mask=mask_tensor)
        return out