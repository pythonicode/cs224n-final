import pandas as pd
import torch
from argparse import ArgumentParser
from models import *
from datasets import *
from transformers import LongformerTokenizerFast
from sklearn.model_selection import train_test_split
import wandb


parser = ArgumentParser()
parser.add_argument('--mode', choices=['train', 'pretrain'])
parser.add_argument('--model', choices=['biglongbirdformer', 'longformer'])
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

LONGFORMER_PATH = 'allenai/longformer-base-4096'
BIGBIRD_PATH = 'google/bigbird-roberta-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    wandb.init(project="cs224n")
    # READ TRAINING DATA
    train_csv = pd.read_csv('./input/feedback-prize-2021/train.csv')
    train_df = pd.read_parquet('./input/parquets/train_all.parquet', engine='pyarrow')

    # SETUP REQUIRED OBJECTS
    if args.model == 'biglongbirdformer':
        tokenizer = LongformerTokenizerFast.from_pretrained(LONGFORMER_PATH)
        model = BigLongBirdFormer(LONGFORMER_PATH, BIGBIRD_PATH, 15, (len(train_df)/args.batch * args.epochs))
        dataset = FeedbackDataset(train_df, tokenizer, max_length=1024, csv=train_csv)
        collate = Collate(tokenizer)

    if args.mode == 'train':
        train_set, valid_set = train_test_split(dataset, test_size=0.1, shuffle=True)
        print(f"{DEVICE} | Train: {len(train_set)} | Validation: {len(valid_set)} | Batch Size {args.batch} | Epochs {args.epochs}")
        model.fit(
            device=DEVICE,
            train_dataset=train_set,
            train_bs=args.batch,
            train_collate_fn=collate,
            train_shuffle=True,
            valid_dataset=valid_set,
            valid_bs=args.batch,
            valid_collate_fn=collate,
            epochs=args.epochs
        )
        model.save('./output/model.bin')