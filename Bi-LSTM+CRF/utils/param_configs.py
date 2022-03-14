import torch
# 配置各类参数
class Configs(object):
    def __init__(self):
        # # 路径类 带*的是运行前的必要文件  未带*文件/文件夹若不存在则训练过程会生成
        # self.train_path = 'data/train'  # *
        # self.dev_path = 'data/test'  # *
        # self.class_ls_path = 'data/class.txt'  # *
        # self.pretrain_dir = '/data/sgns.sogou.char'  # 前期下载的预训练词向量*
        # self.test_path = 'data/test.txt'  # 若该文件不存在会加载dev.txt进行最终测试
        # self.vocab_path = 'data/vocab.pkl'
        # self.model_save_dir = 'checkpoint'
        # self.model_save_name = self.model_save_dir + '/BiLSTM_CRF_faster.ckpt'  # 保存最佳dev acc模型

        # 可调整的参数
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz,  若不存在则后期生成
        # 随机初始化:random
        # self.embedding_type = 'embedding_SougouNews.npz'
        self.use_gpu = True  # 是否使用gpu(有则加载 否则自动使用cpu)
        self.batch_size = 16
        self.num_epochs = 10  # 训练轮数
        self.learning_rate = 0.001  # 训练发现0.001比0.01收敛快(Adam)
        self.embedding_dim = 300  # 词嵌入维度
        self.hidden_size = 300  # 隐藏层维度
        self.num_layers = 1  # RNN层数
        self.bidirectional = True  # 双向 or 单向
        self.padding_size = 100
        # self.require_improvement = 1  # 1个epoch若在dev上acc未提升则自动结束


        # self.class_ls = ["O", "B-BANK", "I-BANK",
        #                  "B-PRODUCT",'I-PRODUCT',
        #                 'B-COMMENTS_N', 'I-COMMENTS_N',
        #                  'B-COMMENTS_ADJ','I-COMMENTS_ADJ']

        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        # 标签：idx
        self.tag2idx = {"O": 0,
                        "B-BANK": 1, "I-BANK": 2,
                        "B-PRODUCT": 3, 'I-PRODUCT': 4,
                        'B-COMMENTS_N': 5, 'I-COMMENTS_N': 6,
                        'B-COMMENTS_ADJ': 7, 'I-COMMENTS_ADJ': 8,
                        START_TAG: 9, STOP_TAG: 10}
        self.idx2tag = {0: "O",
                        1: "B-BANK", 2: "I-BANK",
                        3: "B-PRODUCT", 4: 'I-PRODUCT',
                        5: 'B-COMMENTS_N', 6: 'I-COMMENTS_N',
                        7: 'B-COMMENTS_ADJ', 8: 'I-COMMENTS_ADJ',
                        9: START_TAG, 10: STOP_TAG}

        self.num_class = len(self.tag2idx)
        self.word2idx = None
        self.vocab_len = None  # 词表大小(训练集总的字数(字符级)） 在embedding层作为参数 后期赋值
        self.embedding_pretrained = None  # 根据config.embedding_type后期赋值  random:None  else:tensor from embedding_type
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'