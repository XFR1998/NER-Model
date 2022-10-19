#%%

import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup,BertConfig
from nezha import NeZhaConfig, NeZhaModel, NeZhaForMaskedLM

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

bio_list = [' '.join(i) for i in bio_list]

#%%



#%%

df['bio'] = bio_list


#%%

from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(df, test_size = 0.2, random_state=args.seed)
train_data.index = list(range(len(train_data)))
valid_data.index = list(range(len(valid_data)))
# print(len(train_data), len(valid_data))
print('训练集大小：',len(train_data))
print('验证集大小：',len(valid_data))

#%%


args.tag2idx = {'O':0, 'B-0':1, 'I-0':2}
args.idx2tag = {0: 'O', 1: 'B-0', 2:'I-0'}

#%%



#%%


print('pretrained save_model use'+args.nezha_dir)
tokenizer = BertTokenizer.from_pretrained(args.nezha_dir)

from data_helper import create_data_loader
train_data_loader = create_data_loader(train_data['text'], train_data['bio'], args, tokenizer)
valid_data_loader = create_data_loader(valid_data['text'], valid_data['bio'], args, tokenizer)

#%%

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
    for sample in tqdm(data_loader):
        input_ids = sample['input_ids'].to(args.device)
        attention_mask = sample['attention_mask'].to(args.device)
        label_ids = sample['label_ids'].to(args.device)
        out, loss = model(input_ids=input_ids,
                        label_ids=label_ids,
                        attention_mask=attention_mask)

        train_loss += loss.item()
        loss.backward()

        # -----------------------------------对抗攻击------------------------------------------------
        if args.use_fgm:
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            # loss_adv = model(batch_input, batch_label)
            out, loss_adv = model(input_ids=input_ids,
                        label_ids=label_ids,
                        attention_mask=attention_mask)
            loss_adv = loss_adv.mean()
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

        if args.use_pgd:
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                out, loss_adv = model(input_ids=input_ids,
                                      label_ids=label_ids,
                                      attention_mask=attention_mask)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度

            pgd.restore()


            # ----------------------------------------------------------------------------------------


        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        if args.ema != False:
            args.ema.update()



    return train_loss/len(data_loader)

from ark_nlp.factory.utils.conlleval import get_entity_bio
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
        for sample in tqdm(data_loader):
            input_ids = sample['input_ids'].to(args.device)
            attention_mask = sample['attention_mask'].to(args.device)
            label_ids = sample['label_ids'].to(args.device)
            out, loss = model(input_ids=input_ids,
                        label_ids=label_ids,
                        attention_mask=attention_mask)

            val_loss += loss.item()


            predict_ids = out
            # predict_ids
            #%%
            label_ids = sample['label_ids'].numpy().tolist()

            entity_all_label_ids = []
            entity_all_predict_ids = []
            for i in range(len(label_ids)):
                tmp_label, tmp_predict = [], []
                # 因为我crf有做mask所以这里的len(len(predict_tag[i]))是不带有pad的长度
                for j in range(0, len(predict_ids[i])):
                    tmp_label.append(args.idx2tag[label_ids[i][j]])
                    tmp_predict.append(args.idx2tag[predict_ids[i][j]])
                entity_all_label_ids.append(tmp_label)
                entity_all_predict_ids.append(tmp_predict)


            for label, pred in zip(entity_all_label_ids, entity_all_predict_ids):
                label_entity = return_entity(label)
                pred_entity = return_entity(pred)
                jc_score_list.append(jaccard_score(pred=pred_entity, label=label_entity))

    return val_loss/len(data_loader), np.mean(jc_score_list)


#%%



#%%



#%%



#%%



#%%



#%%

from model import BERT_CRF

if torch.cuda.is_available():
    args.device = 'cuda:0'
    print('使用：', args.device,' ing........')

model = BERT_CRF(args=args).to(args.device)


print('batch_size: ',args.batch_size, 'epochs: ',args.max_epochs)
num_total_steps = len(train_data_loader) * args.max_epochs
from util import build_optimizer, build_optimizer_diff_lr
optimizer, scheduler = build_optimizer(args, model, num_total_steps=num_total_steps)
# optimizer, scheduler = build_optimizer_diff_lr(args, model, num_total_steps=num_total_steps)



if args.ema==True:
    print('-'*10,'采用EMA机制训练','-'*10)
    from tricks import EMA
    args.ema = EMA(model, 0.995)
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
