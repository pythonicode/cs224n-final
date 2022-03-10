import sys
sys.path.append('./input/tez-lib')
import numpy as np
import tez
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from sklearn import metrics
from transformers import AutoModel, get_cosine_schedule_with_warmup

class BigLongBirdFormer(tez.Model):
    def __init__(self, longformer_path, bigbird_path, num_labels, n_steps):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(longformer_path)
        self.bigbird = AutoModel.from_pretrained(bigbird_path)
        self.sequence_drop = nn.Dropout(p=0.2)
        self.output = nn.Linear(self.longformer.config.hidden_size +  self.bigbird.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.n_steps = n_steps

    def monitor_metrics(self, logits, labels):
        if logits is None:
            return {}
        logits = torch.argmax(logits, axis=1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        accuracy = 0
        for i in range(logits.shape[0]):
            accuracy += metrics.accuracy_score(labels[i], logits[i])
        wandb.log({ "accuracy": accuracy })
        return { "accuracy": accuracy }
    
    def fetch_scheduler(self):
        return get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.n_steps)
    
    def fetch_optimizer(self):
        params = [param[1] for param in self.named_parameters()]
        adam = optimizers.AdamW(params, lr=3e-5)
        print(adam.state_dict())
        return adam

    def forward(self, input_ids, attention_mask, labels=None):
        longformer_out = self.longformer(input_ids, attention_mask)
        bigbird_out = self.bigbird(input_ids, attention_mask)
        sequence_output = torch.cat((longformer_out.last_hidden_state, bigbird_out.last_hidden_state), dim=2)
        logits = self.output(self.sequence_drop(sequence_output))
        logits = torch.permute(logits, (0, 2, 1))
        loss = nn.CrossEntropyLoss()(logits, labels)
        wandb.log({"loss": loss})
        metrics = self.monitor_metrics(logits, labels) if labels is not None else None
        return logits, loss, metrics