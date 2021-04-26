import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, ElectraForSequenceClassification, AdamW

class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])


        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        label = item["labels"]

        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.labels)

def tokenized_dataset(csv_path, tokenizer):

    dataset = pd.read_csv(csv_path)
    label = dataset["label"]
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + 'SEPERATE' + e02
        concat_entity.append(temp)
    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=190,
        add_special_tokens=True,
    )

    return tokenized_sentences, label
