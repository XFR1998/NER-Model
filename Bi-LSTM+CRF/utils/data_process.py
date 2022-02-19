from torch.utils.data import Dataset, DataLoader, TensorDataset

class NER_Dataset(Dataset):
    def __init__(self, data=None, configs=None):
        self.data = data
        self.configs = configs
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pass

    def convert_sentence_label_to_idx(self):
        # 首先固定句子长度，短补‘<PAD>’,长截断
        for i in range(len(self.data)):
            if len(self.data[i][0]) < self.configs.padding_size:
                self.data[i][0] += ['<PAD>'] * (self.configs.padding_size - len(self.data[i][0]))
            else:
                self.data[i][0] = self.data[i][0][:self.configs.padding_size]

            # 标签如何处理？怎么填充

def create_data_loader(data, configs):
    ds = NER_Dataset(data=data, configs=configs)
