#%%

import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup,BertConfig
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from ark_nlp.model.ner.global_pointer_bert import Dataset as Dt
from ark_nlp.factory.utils.conlleval import get_entity_bio
from model import GlobalPointer, GlobalPointerNERPredictor, GlobalPointerCrossEntropy
#%%

from config import parse_args

args = parse_args()
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
setup_seed(args.seed)

#%%

data_path = '../dataset/train.csv'
df = pd.read_csv(data_path, delimiter="\t")
df['tag'] = df['tag'].apply(lambda x: eval(x))
df.info()

df.head(5)


#%%

bio_list = []
for i in tqdm(range(len(df))):
    text = df['text'][i]
    tags = df['tag'][i]
    bios = ['O']*len(text)
    for t in tags:
        idx = text.find(t)
        bios[idx] = 'B-0'
        for j in range(idx+1, idx+len(t)):
            bios[j] = 'I-0'
    bio_list.append(bios)

#%%

# bio_list

#%%

df['BIO'] = bio_list
#%%
df['text'][0]
#%%
all_entity_labels = []

for i in range(len(df)):
    entity_labels=[]
    for _type, _start_idx, _end_idx in get_entity_bio(df['BIO'][i], id2label=None):
        entity_labels.append({
            'start_idx': _start_idx,
            'end_idx': _end_idx,
            'type': _type,
            'entity': df['text'][i][_start_idx: _end_idx + 1]
        })
    all_entity_labels.append(entity_labels)
#%%
df['label'] = all_entity_labels
#%%
df = df.drop(columns=['tag'], axis=1)
df.head(5)
#%%
#%%
#%%
#%%
#%%

from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(df, test_size = 0.2, random_state=args.seed)
train_data.index = list(range(len(train_data)))
valid_data.index = list(range(len(valid_data)))
train_data['label'] = train_data['label'].apply(lambda x: str(x))
valid_data['label'] = valid_data['label'].apply(lambda x: str(x))
print('训练集大小：',len(train_data))
print('验证集大小：',len(valid_data))

#%%


# args.tag2idx = {'O':0, 'B-0':1, 'I-0':2}
# args.idx2tag = {0: 'O', 1: 'B-0', 2:'I-0'}

#%%
label_list = ['0', 'O']


train_dataset = Dt(train_data, categories=label_list)
dev_dataset = Dt(valid_data, categories=label_list)
print('pretrained save_model use'+args.bert_dir)
tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=54)

train_dataset.convert_to_ids(ark_tokenizer)
dev_dataset.convert_to_ids(ark_tokenizer)


Ent2id = train_dataset.cat2id
id2Ent = train_dataset.id2cat

#%%
Ent2id, id2Ent
#%%
from data_helper import create_data_loader
train_data_loader, valid_data_loader = create_data_loader(train_data_df=train_data,
                                                          dev_data_df=valid_data,
                                                          ark_tokenizer=ark_tokenizer,
                                                          args=args,
                                                          train_dataset=train_dataset,
                                                          dev_dataset=dev_dataset,
                                                          bert_tokenizer=tokenizer)

print(len(train_data_loader), len(valid_data_loader))

#%%



#%%

def jaccard_score(pred, label):
    return len(set(pred) & set(label)) / len(set(pred) | set(label))

#%%



#%%
def train_epoch(model, data_loader, optimizer, args, scheduler):
    # 训练模式
    model = model.train()
    train_loss = 0
    for token_id, at_mask, label_id, token_type_ids in tqdm(data_loader):
        outputs = model(token_id.to(args.device), at_mask.to(args.device), token_type_ids.to(args.device))
        loss = loss_fn(outputs, label_id.to(args.device))
        train_loss+=loss.item()
        loss.backward()

        # -----------------------------------对抗攻击------------------------------------------------
        if args.use_fgm:
            fgm.attack()
            outputs = model(token_id.to(args.device), at_mask.to(args.device), token_type_ids.to(args.device))
            loss_fgm = loss_fn(outputs, label_id.to(args.device)).mean()
            loss_fgm.backward()
            fgm.restore()
        if args.use_pgd:
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                outputs = model(token_id.to(args.device), at_mask.to(args.device), token_type_ids.to(args.device))
                loss_pgd = loss_fn(outputs, label_id.to(args.device)).mean()
                loss_pgd.backward()
            pgd.restore()
            # ----------------------------------------------------------------------------------------




        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        if args.ema != False:
            args.ema.update()



    return train_loss/len(data_loader)


def return_entity(label):
    entity_labels = []
    for _type, _start_idx, _end_idx in get_entity_bio(label, id2label=None):
            entity_labels.append({
                'start_idx': _start_idx,
                'end_idx': _end_idx,
                'type': _type
            })
    entity_labels = [str(dic['start_idx'])+'-'+str(dic['end_idx']) for dic in entity_labels]
    return entity_labels


def eval_epoch(model, data_loader, args):
    # 验证模式
    model = model.eval()
    if args.ema!=False:
        args.ema.apply_shadow()
    val_loss = 0
    jc_score_list = []
    # 关闭自动求导，省内存加速，因为是不是训练模式了，没必要求导
    with torch.no_grad():
        for token_id, at_mask, label_id, token_type_ids in tqdm(data_loader):
            outputs = model(token_id.to(args.device), at_mask.to(args.device), token_type_ids.to(args.device))
            loss = loss_fn(outputs, label_id.to(args.device))
            val_loss += loss.item()


            y_pred = outputs
            y_true = label_id.to(args.device)
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            pred = []
            true = []
            for b, l, start, end in zip(*np.where(y_pred > 0)):
                pred.append((b, l, start, end))
            for b, l, start, end in zip(*np.where(y_true > 0)):
                true.append((b, l, start, end))

            jc_score_list.append(jaccard_score(pred=pred, label=true))

    return val_loss/len(data_loader), np.mean(jc_score_list)

#%%

if torch.cuda.is_available():
    args.device = 'cuda:0'
    print('使用：', args.device,' ing........')


model = GlobalPointer(args, len(Ent2id), 64).to(args.device)  # (encoder, ent_type_size, inner_dim)


print('batch_size: ',args.batch_size, 'epochs: ',args.max_epochs)
num_total_steps = len(train_data_loader) * args.max_epochs
from util import build_optimizer
optimizer, scheduler = build_optimizer(args, model, num_total_steps=num_total_steps)
loss_fn = GlobalPointerCrossEntropy().to(args.device)



if args.ema==True:
    print('-'*10,'采用EMA机制训练','-'*10)
    from tricks import EMA
    args.ema = EMA(model, 0.999)
    args.ema.register()

if args.use_fgm==True:
    print('-' * 10, '采用FGM对抗训练', '-' * 10)
    from tricks import FGM
    # 初始化
    fgm = FGM(model)

if args.use_pgd==True:
    print('-' * 10, '采用PGD对抗训练', '-' * 10)
    from tricks import PGD
    # 初始化
    pgd = PGD(model=model)
    K = 3

#%%



#%%

best_jc_score = 0
for epoch in range(args.max_epochs):
    print('——'*10, f'Epoch {epoch + 1}/{args.max_epochs}', '——'*10)
    train_loss = train_epoch(model, train_data_loader, optimizer, args, scheduler)
    # #scheduler.step()
    # print('-'*20)
    print(f'Train loss : {round(train_loss, 2)}\n')
    val_loss, jc_score = eval_epoch(model, valid_data_loader, args)



    if jc_score>best_jc_score:
        best_jc_score = jc_score
        print(f'val loss : {round(val_loss, 3)}')
        print(f"jc_score: {round(jc_score, 3)}")
        print('-'*20)
        torch.save(model.state_dict(), './save_model/best_model.pth')
        print('+'*6,'best save_model saved','+'*6)

    if args.ema != False:
        args.ema.restore()