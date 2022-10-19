import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")
    parser.add_argument('-f', type=str, default="读取额外的参数")
    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================

    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--max_len', default=50, type=int, help="text max len")
    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=32, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=16, help='How many epochs')
    parser.add_argument('--max_steps', default=17000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")





    # ========================== My tricks =============================
    parser.add_argument('--ema', type=bool, default=False, help="whether use ema or not")
    parser.add_argument('--use_fgm', type=bool, default=False, help="whether use fgm or not")
    parser.add_argument('--use_pgd', type=bool, default=False, help="whether use pgd or not")



    return parser.parse_args()
