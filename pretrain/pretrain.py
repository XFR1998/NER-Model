#%%
import torch
import random
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import warnings
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

warnings.filterwarnings('ignore')
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
setup_seed(2022)
#%%
# train_data = pd.read_csv('../dataset/train.csv', delimiter="\t")
# test_data = pd.read_csv('../dataset/test.csv', sep='\t')
# train_data.info()
# test_data.info()
#
# data = pd.concat([train_data, test_data])
# data.info()
#%%
# from sklearn.model_selection import train_test_split
# train_data, valid_data = train_test_split(data, test_size = 0.2, random_state=4)
# train_data.index = list(range(len(train_data)))
# valid_data.index = list(range(len(valid_data)))
# # print(len(train_data), len(valid_data))
# print('训练集大小：',len(train_data))
# print('验证集大小：',len(valid_data))
#%%
# train_text = '\n'.join(train_data.text.tolist())
# valid_text = '\n'.join(valid_data.text.tolist())
# with open('./data_train.txt', 'w', encoding='utf-8') as f:
#     f.write(train_text)
# with open('./data_valid.txt', 'w', encoding='utf-8') as f:
#     f.write(valid_text)
#%%
model_name = '../hfl/chinese-roberta-wwm-ext'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./my_pretrain_models')




train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data_train.txt",  # mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data_valid.txt",  # mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./my_pretrain_models_chk",  # select save_model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
trainer.save_model(f'./my_pretrain_models')