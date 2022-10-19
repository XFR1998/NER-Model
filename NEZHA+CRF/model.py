import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel,BertConfig
from nezha import NeZhaConfig, NeZhaModel, NeZhaForMaskedLM


class BERT_CRF(nn.Module):
    def __init__(self, args=None):
        super(BERT_CRF, self).__init__()
        self.args = args
        self.tag2idx = args.tag2idx
        bert_config = NeZhaConfig.from_json_file(self.args.bert_dir + '/config.json')
        bert_config.num_labels = len(self.tag2idx)
        self.bert = NeZhaModel.from_pretrained(self.args.bert_dir, config=bert_config)




        # 转换参数矩阵 输入i,j是得分从j转换到i
        self.tagset_size = len(self.tag2idx)
        # 将lstm的输出映射到标记空间
        # (B, tagset_size)  因为crf层需要加上<START> <END>这两个预测
        # self.dropout = nn.Dropout(0.7)
        # self.fc1 = nn.Linear(768, 256)
        # self.tanh = nn.Tanh()
        # self.fc2 = nn.Linear(256, self.tagset_size)
        self.classifier = nn.Linear(768, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)


    def neg_log_likelihood(self, logits=None, label_ids=None, attention_mask=None):  # 损失函数

        return -self.crf(emissions=logits, tags=label_ids, mask=attention_mask)



    def forward(self, input_ids=None, label_ids=None, attention_mask=None):

        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                               attention_mask=attention_mask)
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if label_ids!=None:
            out = self.crf.decode(emissions=logits,
                                  mask=attention_mask)
            loss = self.neg_log_likelihood(logits=logits,
                               label_ids=label_ids,
                               attention_mask=attention_mask)
            return out, loss
        else:
            out = self.crf.decode(emissions=logits,
                                  mask=attention_mask)
            return out



