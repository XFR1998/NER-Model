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
        nezha_config = NeZhaConfig.from_json_file(self.args.nezha_dir + '/config.json')
        nezha_config.num_labels = len(self.tag2idx)
        self.nezha = NeZhaModel.from_pretrained(self.args.nezha_dir, config=nezha_config)

        roberta_config = BertConfig.from_pretrained(self.args.roberta_dir + '/config.json')
        self.roberta = BertModel.from_pretrained(self.args.roberta_dir, config=roberta_config)



        # 转换参数矩阵 输入i,j是得分从j转换到i
        self.tagset_size = len(self.tag2idx)
        self.classifier = nn.Linear(768, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)


    def neg_log_likelihood(self, logits=None, label_ids=None, attention_mask=None):  # 损失函数

        return -self.crf(emissions=logits, tags=label_ids, mask=attention_mask)



    def forward(self, input_ids=None, label_ids=None, attention_mask=None):

        nezha_sequence_output, nezha_pooled_output = self.nezha(input_ids=input_ids,
                                               attention_mask=attention_mask)

        roberta_sequence_output, roberta_pooled_output = self.roberta(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   return_dict=False)


        sequence_output = (nezha_sequence_output+roberta_sequence_output) / 2

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



