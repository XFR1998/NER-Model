#%%

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertTokenizer

#%%

data_path = '../dataset/test.csv'
df = pd.read_csv(data_path, delimiter="\t")
df.info()
df.head(5)



#%%

from config import parse_args

args = parse_args()
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
setup_seed(args.seed)


args.tag2idx = {'O':0, 'B-0':1, 'I-0':2}
args.idx2tag = {0: 'O', 1: 'B-0', 2:'I-0'}

#%%

test_data = df['text']

#%%

# test_data



#%%



#%%

from data_helper import NER_Dataset
print('pretrained save_model use'+args.nezha_dir)
tokenizer = BertTokenizer.from_pretrained(args.nezha_dir)
test_data_datsset = NER_Dataset(text=test_data, args=args, test_mode=True, tokenizer=tokenizer)
sampler = SequentialSampler(test_data_datsset)
dataloader = DataLoader(test_data_datsset,
                        batch_size=args.test_batch_size,
                        sampler=sampler)



#%%

if torch.cuda.is_available():
    args.device = 'cuda:0'
    print('使用：', args.device,' ing........')
from model import BERT_CRF
model = BERT_CRF(args)
path = f'./save_model/best_model.pth'
model.load_state_dict(torch.load(path, map_location='cpu'))
model=model.to(args.device)

#%%

# 保存有所样本的预测结果
predict_tag = []

model = model.eval()
with torch.no_grad():
    for sample in tqdm(dataloader, 'val'):
        input_ids = sample['input_ids'].to(args.device)
        attention_mask = sample['attention_mask'].to(args.device)
        # label_tensor = sample['label_tensor'].to(configs.device)
        out = model(input_ids=input_ids,
                    label_ids=None,
                    attention_mask=attention_mask)

        for l in out:
            temp = []
            for i in l:
                temp.append(args.idx2tag[i])
            predict_tag.append(temp)

#%%

from ark_nlp.factory.utils.conlleval import get_entity_bio
def extract_entity(label, text):
    entity_labels = []
    for _type, _start_idx, _end_idx in get_entity_bio(label, id2label=None):
        # 因为传入bert时是前面加了[CLS]的，所以这里索引要减1
        _start_idx = _start_idx - 1
        _end_idx = _end_idx - 1
        entity_labels.append({
            'start_idx': _start_idx,
            'end_idx': _end_idx,
            'type': _type,
            'entity': text[_start_idx: _end_idx + 1]
        })
    entity_list = []
    for info in entity_labels:
        entity_list.append(info['entity'])
    return entity_list

#%%

# extract_entity(label=predict_tag[0], text=df['text'][0])

#%%

tag_list = []
for idx, label in enumerate(predict_tag):
    tag_list.append(extract_entity(label=label, text=df['text'][idx]))

#%%

tag_list

#%%

new_df = pd.DataFrame({'tag': tag_list})
new_df.to_csv('suubmit.csv', index=False)