import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class NER_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, ark_tokenizer, max_len, ark_data, bert_tokenizer):
        self.len = len(dataframe)
        self.data = dataframe
        self.ark_tokenizer = ark_tokenizer
        self.max_len = max_len
        self.ark_data = ark_data
        self.bert_tokenizer = bert_tokenizer

    def __getitem__(self, index):
        row = self.data.iloc[index]
        token_ids, at_mask = self.get_token_ids(row)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(at_mask, dtype=torch.long), torch.tensor(
            self.ark_data[index]['label_ids'].to_dense()), torch.tensor(self.ark_data[index]['token_type_ids'],
                                                                        dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def get_token_ids(self, row):
        sentence = row.text
        tokens = self.ark_tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        padding = [0] * (self.max_len - len(tokens))
        at_mask = [1] * len(tokens)
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids + padding
        at_mask = at_mask + padding
        return token_ids, at_mask

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        at_mask = torch.stack([x[1] for x in batch])
        labels = torch.stack([x[2] for x in batch])
        token_type_ids = torch.stack([x[3] for x in batch])
        return token_ids, at_mask, labels.squeeze(), token_type_ids





def create_data_loader(train_data_df=None, dev_data_df=None, ark_tokenizer=None, args=None, train_dataset=None, dev_dataset=None, bert_tokenizer=None):
    ner_train_dataset = NER_Dataset(train_data_df, ark_tokenizer, args.max_len, train_dataset, bert_tokenizer)
    ner_dev_dataset = NER_Dataset(dev_data_df, ark_tokenizer, args.max_len, dev_dataset, bert_tokenizer)

    train_loader = DataLoader(ner_train_dataset,  # 1250
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=ner_train_dataset.collate_fn)
    valid_loader = DataLoader(ner_dev_dataset,  # 13
                            batch_size=args.val_batch_size,
                            shuffle=True,
                            collate_fn=ner_dev_dataset.collate_fn)

    return train_loader, valid_loader