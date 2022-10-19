import torch
from torch.utils.data import Dataset, DataLoader

class NER_Dataset(Dataset):
    def __init__(self, data=None, args=None, test_mode=False):

        self.args = args
        self.test_mode = test_mode
        self.data, self.data_tensor = self.convert_sentence_label_to_idx(data) # 将data转成tensor



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = {}


        if not self.test_mode:
            sample['sentence'] = self.data[item][0]
            sample['sentence_tensor'] = self.data_tensor[item][0]
            sample['mask_tensor'] = self.data_tensor[item][2]
            sample['label'] = self.data[item][1]
            sample['label_tensor'] = self.data_tensor[item][1]
        else:
            sample['sentence'] = self.data[item][0]
            sample['sentence_tensor'] = self.data_tensor[item][0]
            sample['mask_tensor'] = self.data_tensor[item][1]
        return sample

    def convert_sentence_label_to_idx(self, data):
        # 首先固定句子长度，短补‘<PAD>’,长截断
        contents = []
        tensor_contents = []
        # data[i][0]是sentence分字后的列表
        # data[i][1]是对应各字的标签
        for i in range(len(data)):

            len_data = len(data[i][0])

            if len_data < self.args.max_len:
                if not self.test_mode:
                    sent = data[i][0] + (['<PAD>'] * (self.args.max_len - len_data))
                    label = data[i][1] + (['O'] * (self.args.max_len - len_data)) # 标签的话，标什么应该都问题不大，因为后面传入CRF会mask掉
                    contents.append((' '.join(sent), ' '.join(label)))

                    # 转tensor，加个mask(CRF所需)
                    sent_tensor = torch.tensor([self.args.word2idx[w] if w in self.args.word2idx.keys() else self.args.word2idx['<UNK>'] for w in sent])
                    label_tensor = torch.tensor([self.args.tag2idx[t] for t in label])

                    mask_tensor = torch.tensor([1]*len_data + [0]*(self.args.max_len - len_data)).bool()

                    tensor_contents.append((sent_tensor, label_tensor, mask_tensor))
                else:
                    sent = data[i][0] + (['<PAD>'] * (self.args.max_len - len_data))
                    contents.append(' '.join(sent))

                    # 转tensor，加个mask(CRF所需)
                    sent_tensor = torch.tensor(
                        [self.args.word2idx[w] if w in self.args.word2idx.keys() else self.args.word2idx['<UNK>'] for w
                         in sent])

                    mask_tensor = torch.tensor([1] * len_data + [0] * (self.args.max_len - len_data)).bool()

                    tensor_contents.append((sent_tensor, mask_tensor))

            else:
                if not self.test_mode:
                    sent = data[i][0][:self.args.max_len]
                    label = data[i][1][:self.args.max_len]
                    contents.append((' '.join(sent), ' '.join(label)))

                    # 转tensor，加个mask(CRF所需)
                    sent_tensor = torch.tensor([self.args.word2idx[w] if w in self.args.word2idx.keys() else self.args.word2idx['<UNK>'] for w in sent])
                    label_tensor = torch.tensor([self.args.tag2idx[t] for t in label])

                    mask_tensor = torch.tensor([1] * self.args.max_len).bool()

                    tensor_contents.append((sent_tensor, label_tensor, mask_tensor))
                else:
                    sent = data[i][0][:self.args.max_len]
                    contents.append(' '.join(sent))

                    # 转tensor，加个mask(CRF所需)
                    sent_tensor = torch.tensor(
                        [self.args.word2idx[w] if w in self.args.word2idx.keys() else self.args.word2idx['<UNK>'] for w
                         in sent])

                    mask_tensor = torch.tensor([1] * self.args.max_len).bool()

                    tensor_contents.append((sent_tensor, mask_tensor))

        return contents, tensor_contents

def create_data_loader(data, args, test_mode=False):
    ds = NER_Dataset(data=data, args=args, test_mode=test_mode)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True)