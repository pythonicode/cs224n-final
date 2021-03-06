import pandas as pd
import torch
from argparse import ArgumentParser
from models import *
from datasets import *
from transformers import LongformerTokenizerFast, DataCollatorForLanguageModeling, LongformerForMaskedLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import wandb
import os


parser = ArgumentParser()
parser.add_argument('--mode', choices=['train', 'pretrain'])
parser.add_argument('--model', choices=['biglongbirdformer', 'longformer'], default='biglongbirdformer')
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

    essays = []
    for path in os.listdir('./input/argumentative_essays'):
        with open(path) as f:
            if path.endswith('.txt'):
                essays.append(f.read())
    
    pretrain_df = pd.DataFrame(essays, columns=['text'])

    tokenizer = LongformerTokenizerFast.from_pretrained(LONGFORMER_PATH)
    if args.model == 'biglongbirdformer':
        model = BigLongBirdFormer(LONGFORMER_PATH, BIGBIRD_PATH, 15, (len(train_df)/args.batch * args.epochs))
        
    if args.mode == 'train':
        dataset = FeedbackDataset(train_df, tokenizer, max_length=1024, csv=train_csv)
        collate = FeedbackCollate(tokenizer)
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
    
    if args.mode == 'pretrain':
        model = LongformerForMaskedLM.from_pretrained(LONGFORMER_PATH)
        dataset = MaskingDataset(pretrain_df)
        collate = DataCollatorForLanguageModeling(tokenizer)
        training_args = TrainingArguments(output_dir='./output')
        trainer = Trainer(model, training_args, collate, train_dataset=dataset)