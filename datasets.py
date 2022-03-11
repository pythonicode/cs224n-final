import torch
from torch.utils.data.dataset import Dataset

class MaskingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        output = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return output

class FeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, csv):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.csv = csv
        self.label_to_index = {
            'O': 0,
            'B-Lead': 1,
            'I-Lead': 2,
            'B-Position': 3,
            'I-Position': 4,
            'B-Claim': 5,
            'I-Claim': 6,
            'B-Counterclaim': 7,
            'I-Counterclaim': 8,
            'B-Rebuttal': 9,
            'I-Rebuttal': 10,
            'B-Evidence': 11,
            'I-Evidence': 12,
            'B-Concluding Statement': 13,
            'I-Concluding Statement': 14,
            'PAD': -100,
        }

        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]
        inputs = self.tokenizer(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        sub_label = self.csv['discourse_type'][idx]
        labels = [self.label_to_index["B-" + sub_label]] + [self.label_to_index["I-" + sub_label]] * (len(inputs['input_ids']) - 1)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        } 

### THIS COLLATE FUNCTION MUST BE USED WITH ABOVE DATASET TO ENSURE PROPER FORMATTING

class FeedbackCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()

        output['input_ids'] = [sample['input_ids'] for sample in batch]
        output['attention_mask'] = [sample['attention_mask'] for sample in batch]
        output['labels'] = [sample['labels'] for sample in batch]
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output['input_ids']])
        
        # add padding
        output['input_ids'] = [sample + (batch_max-len(sample)) * [self.tokenizer.pad_token_id] for sample in output['input_ids']]
        output['attention_mask'] = [sample + (batch_max-len(sample)) * [0] for sample in output['attention_mask']]
        output['labels'] = [sample + (batch_max-len(sample)) * [-100] for sample in output['labels']]
        
        # convert to tensors
        output['input_ids'] = torch.tensor(output['input_ids'], dtype=torch.long)
        output['attention_mask'] = torch.tensor(output['attention_mask'], dtype=torch.long)
        output['labels'] = torch.tensor(output['labels'], dtype=torch.long)
    
        return output
