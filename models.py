import tez
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup

class BigLongBirdFormer(tez.Model):
    def __init__(self, longformer_path, bigbird_path, num_labels, n_steps, train=False):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(longformer_path)
        self.bigbird = AutoModel.from_pretrained(bigbird_path)
        self.output = nn.Linear(self.longformer.config.hidden_size +  self.bigbird.config.hidden_size, num_labels)
        self.training = train
        self.n_steps = n_steps
        self.num_labels = num_labels

    def monitor_metrics(self, logits, labels):
        if logits is None:
            return {}
        logits = torch.sigmoid(logits).cpu().detach().numpy() >= 0.8
        labels = F.one_hot(labels, num_classes=self.num_labels).cpu().detach().numpy()
        print(logits.shape, labels.shape)
        accuracy = metrics.accuracy_score(labels, logits)
        wandb.log({ "accuracy": accuracy })
        return { "accuracy": accuracy }
    
    def fetch_scheduler(self):
        return get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.n_steps
        )
    
    def fetch_optimizer(self):
        params = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimize = [
            {
                "params": [
                    p for n, p in params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimize) 

    def forward(self, input_ids, attention_mask, labels=None):
        longformer_out = self.longformer(input_ids, attention_mask)
        bigbird_out = self.bigbird(input_ids, attention_mask)
        sequence_output = torch.cat((longformer_out.last_hidden_state, bigbird_out.last_hidden_state), dim=2)
        logits = self.output(sequence_output)
        logits = torch.permute(logits, (0, 2, 1))
        loss = nn.CrossEntropyLoss()(logits, labels) if self.training else 0
        wandb.log({"loss": loss})
        metrics = self.monitor_metrics(logits[:, :, 0], labels[:, 0]) if self.training else 0
        return logits, loss, metrics