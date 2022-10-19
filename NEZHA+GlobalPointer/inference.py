#%%

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertTokenizer
from model import GlobalPointer, GlobalPointerNERPredictor
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



#%%

# test_data


from ark_nlp.model.ner.global_pointer_bert import Tokenizer
tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=128)
import os
from ark_nlp.factory.utils.conlleval import get_entity_bio



Ent2id, id2Ent = {'0': 0, 'O': 1}, {0: '0', 1: 'O'}



#%%

if torch.cuda.is_available():
    args.device = 'cuda:0'
    print('使用：', args.device,' ing........')

model = GlobalPointer(args, len(Ent2id), 64).to(args.device)  #
path = f'./save_model/best_model.pth'
model.load_state_dict(torch.load(path, map_location='cpu'))
model=model.to(args.device)

#%%
model.eval()
ner_predictor_instance = GlobalPointerNERPredictor(model, ark_tokenizer, Ent2id, tokenizer)

from tqdm import tqdm

predict_results = []

for i in tqdm(range(len(df))):
    _line = df['text'][i]
    label = len(_line) * ['O']
    for _preditc in ner_predictor_instance.predict_one_sample(_line):
        if 'I' in label[_preditc['start_idx']]:
            continue
        if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
            continue
        if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
            continue

        label[_preditc['start_idx']] = 'B-' + _preditc['type']
        label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
            'start_idx']) * [('I-' + _preditc['type'])]

    predict_results.append(label)

#%%
print(len(predict_results))
#%

#%%
from ark_nlp.factory.utils.conlleval import get_entity_bio
def extract_entity(label, text):
    entity_labels = []
    for _type, _start_idx, _end_idx in get_entity_bio(label, id2label=None):
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
tag_list = []
for idx, label in enumerate(predict_results):
    tag_list.append(extract_entity(label=label, text=df['text'][idx]))

#%%
tag_list
#%%

new_df = pd.DataFrame({'tag': tag_list})
new_df.to_csv('suubmit.csv', index=False)