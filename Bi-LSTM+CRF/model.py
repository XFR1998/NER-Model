import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        self.args = args
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(configs.embedding_pretrained,
        #                                                   freeze=False)  # 表示训练过程词嵌入向量会更新
        # else:

        self.embedding = nn.Embedding(args.vocab_len, embedding_dim=300,
                                      padding_idx=args.word2idx['<PAD>'])  # PAD索引填充


        self.rnn = nn.LSTM(input_size=300,
                           hidden_size=300,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)

        self.tag2idx = args.tag2idx

        # 转换参数矩阵 输入i,j是得分从j转换到i
        self.tagset_size = len(self.tag2idx)
        # 将lstm的输出映射到标记空间
        self.hidden2tag = nn.Linear(300*2, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)


    def _get_lstm_features(self, x):
        # x.shape: (bs, num_words)
        x = self.embedding(x)
        # x.shape: (bs, num_words, embedding_dim)
        bs = x.size(0)
        h_0, c_0 = torch.randn(2, bs,  300).to(x.device), torch.randn(2, bs, 300).to(x.device)
        out, (hidden, c) = self.rnn(x,(h_0, c_0))
        # out.shape: (bs, num_words, hidden_size*2)
        out = self.hidden2tag(out)  # (B,num_directions*hidden_size) -> (B, num_class)
        # out.shape: (bs, num_words, tagset_size)
        return out

    def neg_log_likelihood(self, sentence_tensor=None, label_tensor=None, mask_tensor=None):  # 损失函数
        feats = self._get_lstm_features(sentence_tensor)
        return -self.crf(emissions=feats, tags=label_tensor, mask=mask_tensor)
        # return -self.crf(emissions=feats, tags=label_tensor)


    def forward(self, sentence_tensor=None, label_tensor=None, mask_tensor=None):
        # 数据预处理时，x被处理成是一个tuple,其内容是: (word, label).
        # x:b_size
        # print(sentence_tensor)
        # print(mask_tensor)
        lstm_feats = self._get_lstm_features(sentence_tensor)  # 获取BiLSTM的emission分数

        # Returns: List of list containing the best tag sequence for each batch.
        # 返回列表组成的标签

        if label_tensor!=None:
            out = self.crf.decode(emissions=lstm_feats,
                                  mask=mask_tensor)
            loss = self.neg_log_likelihood(sentence_tensor=sentence_tensor,
                               label_tensor=label_tensor,
                               mask_tensor=mask_tensor)

            return out, loss
        else:
            out = self.crf.decode(emissions=lstm_feats,
                                  mask=mask_tensor)

            return out



