import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class NER_Dataset(Dataset):
    def __init__(self, text=None, label=None, args=None, tokenizer=None, test_mode=False):
        self.text = text.tolist()

        self.data_dict = defaultdict(list)
        self.args = args
        self.test_mode = test_mode
        if not test_mode:
            self.label = label.tolist()
            for i in range(len(self.text)):
                self.build_bert_inputs_train(inputs=self.data_dict,
                                       sentence=self.text[i],
                                       label=self.label[i],
                                       tokenizer=tokenizer,
                                       args=self.args)
        else:
            for i in range(len(self.text)):
                self.build_bert_inputs_test(inputs=self.data_dict,
                                       sentence=self.text[i],
                                       tokenizer=tokenizer,
                                       args=self.args)


    def build_bert_inputs_train(self, inputs, sentence, label, tokenizer,args):
        token_list = list(sentence)
        label_list = label.split(' ')
        assert len(token_list) == len(label_list)

        tokens, labels = [], []
        for i, word in enumerate(token_list):

            if word == ' ' or word == '':
                word = '-'

            token = tokenizer.tokenize(word)

            if len(token) > 1:
                token = [tokenizer.unk_token]

            tokens.extend(token)
            labels.append(label_list[i])

        assert len(tokens) == len(labels)

        inputs_dict = tokenizer.encode_plus(tokens,
                                            add_special_tokens=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            max_length=args.max_len,
                                            pad_to_max_length=True,
                                            truncation=False,
                                            return_tensors='pt'
                                            )

        input_ids = inputs_dict['input_ids'][0] #它返回多了一维，去掉外层那维即可
        # token_type_ids = inputs_dict['token_type_ids']
        attention_mask = inputs_dict['attention_mask'][0].bool()

        label_ids = []
        label_ids.extend([args.tag2idx['O']]) # [CLS]
        label_ids.extend([args.tag2idx[i] for i in labels])
        label_ids.extend([args.tag2idx['O']]) # [SEP]
        label_ids.extend([args.tag2idx['O']]*(args.max_len-torch.sum(inputs_dict['attention_mask']))) # 为padding补标签

        assert len(input_ids) == len(label_ids)

        inputs['input_ids'].append(input_ids)
        inputs['attention_mask'].append(attention_mask)
        inputs['label_ids'].append(label_ids)

    def build_bert_inputs_test(self, inputs, sentence, tokenizer, args):
        token_list = list(sentence)

        tokens, labels = [], []
        for i, word in enumerate(token_list):

            if word == ' ' or word == '':
                word = '-'

            token = tokenizer.tokenize(word)

            if len(token) > 1:
                token = [tokenizer.unk_token]

            tokens.extend(token)


        inputs_dict = tokenizer.encode_plus(tokens,
                                            add_special_tokens=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            max_length=args.max_len,
                                            pad_to_max_length=True,
                                            truncation=False,
                                            return_tensors='pt'
                                            )

        input_ids = inputs_dict['input_ids'][0] #它返回多了一维，去掉外层那维即可
        # token_type_ids = inputs_dict['token_type_ids']
        attention_mask = inputs_dict['attention_mask'][0].bool()

        inputs['input_ids'].append(input_ids)
        inputs['attention_mask'].append(attention_mask)



    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        sample = {}
        sample['text'] = self.text[item]
        sample['input_ids'] = self.data_dict['input_ids'][item]
        sample['attention_mask'] = self.data_dict['attention_mask'][item]

        if not self.test_mode:
            sample['label_ids'] = torch.tensor(self.data_dict['label_ids'][item])
        return sample



def create_data_loader(text=None, label=None, args=None, tokenizer=None):
    ds = NER_Dataset(text=text,
                     label=label,
                     args=args,
                     tokenizer=tokenizer,
                     test_mode=False)

    return DataLoader(ds, batch_size=args.batch_size,shuffle=False)