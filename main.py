import argparse
import time
import warnings

import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from madgrad import MADGRAD

from my_model import Mymodel
from config import Config
from data_set import *

from adamp import AdamP
from focalloss import *

def get_config():
    parser = argparse.ArgumentParser(description="use huggingface models")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--model_name", default="xlm-roberta-large", type=str)
    parser.add_argument("--add_vocab", default=False, type=bool)
    parser.add_argument("--add_lstm", default=False, type=bool)
    parser.add_argument("--mask", default=False, type=bool)
    parser.add_argument("--switch", default=False, type=bool)
    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MODEL_NAME=args.model_name,
        ADD_VOCAB=args.add_vocab,
        ADD_LSTM=args.add_lstm,
        MASK=args.mask,
        SWITCH=args.switch
    )

    return config


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data(config):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if config.ADD_VOCAB:
        tokenizer.add_special_tokens({"additional_special_tokens":["<ent>", "</ent>", "<ent2>", "</ent2>"]})
        # add_vocab = pd.read_csv("/opt/ml/input/data/vocab.csv")
        # add_vocab_list = list(add_vocab["vocab"].values)
        # tokenizer.add_tokens(add_vocab_list)
        print(len(tokenizer.vocab))

    train_token, train_label = tokenized_dataset("/opt/ml/input/data/train/my_train.csv", tokenizer)
    test_token, label = tokenized_dataset("/opt/ml/input/data/test/my_test.csv", tokenizer)

    train_set = RE_Dataset(train_token, train_label)
    test_set = RE_Dataset(test_token, label)

    return train_set, test_set, tokenizer


def get_model(config):
    if config.ADD_LSTM:
        network = Mymodel(config.MODEL_NAME)
    else:
        network = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=42, hidden_dropout_prob=0.0).to(config.device)

    if config.ADD_VOCAB:
        network.resize_token_embeddings(250006)
    optimizer = AdamW(network.parameters(), lr=config.LEARNING_RATE, weight_decay=0.1)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = FocalLoss(gamma=0.5).to(config.device)

    return network, optimizer, scaler, scheduler, criterion


def test_one_epoch(epoch, model, loss_fn, test_loader, device):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, labels) in pbar:
        labels = labels.to(device)

        preds = model(input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
        targets_all += [labels.detach().cpu().numpy()]

        loss = loss_fn(preds, labels)

        loss_sum += loss.item()*labels.shape[0]
        sample_num += labels.shape[0]
    
        description = f"epoch {epoch} loss: {loss_sum/sample_num:.4f}"
        pbar.set_description(description)

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()

    print("     test accuracy = {:.4f}".format(accuracy))

    return accuracy


def train_one_epoch(epoch, model, loss_fn, optim, scaler, train_loader, device, batch_size, tokenizer, config, scheduler=None):
    model.train()

    t = time.time()
    running_loss = 0
    sample_num = 0
    preds_all = []
    targets_all = []            

    change_mask_prop = 0.8
    switch_prop = 0.5

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, labels) in pbar:
        labels = labels.to(device)
        optim.zero_grad()

        with autocast():
            if config.SWITCH:
                switch_p = random.random()
            if config.MASK:
                mask_p = random.random()

            if config.SWITCH and switch_p < switch_prop:
                input_ids = switch_sentence(input_ids, tokenizer)

            if config.MASK and mask_p < change_mask_prop:
                input_ids = custom_to_mask(input_ids, batch_size)
            
            preds = model(input_ids.to(device), attention_mask=attention_mask.to(device))[0]
            loss = loss_fn(preds, labels)

            preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
            targets_all += [labels.detach().cpu().numpy()]
            scaler.scale(loss).backward()

            running_loss += loss.item()*labels.shape[0]
            sample_num += labels.shape[0]

            scaler.step(optim)
            scaler.update()

            description = f"epoch {epoch} loss: {running_loss/sample_num: .4f}"
            pbar.set_description(description)
    
    
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()
    print("     train accuracy = {:.4f}".format(accuracy))

    if scheduler is not None:
        scheduler.step()


def custom_to_mask(input_ids, batch_size):
    for i in range(len(input_ids)):
        sep_idx = np.where(input_ids[i].numpy() == 2)

        mask_idxs = set()
        while len(mask_idxs) <= 3:
            mask_idxs.add(random.randrange(sep_idx[0][1] + 1, sep_idx[0][2]))

        for mask_idx in list(mask_idxs):
            input_ids[i][mask_idx] = 250001
    
    return input_ids


def switch_sentence(input_ids, tokenizer):
    for i in range(len(input_ids)):
        sep_idx = np.where(input_ids[i].numpy() == 2)[0]

        sentence = input_ids[i][sep_idx[1] + 1 : sep_idx[2]]

        decode_sentence = tokenizer.decode(sentence)
        sentence_list = decode_sentence.split(" ")
        switch_idx = random.randrange(len(sentence_list) // 3, len(sentence_list) // 3 * 2)
        
        switch_sentence = " ".join(sentence_list[switch_idx :] + sentence_list[:switch_idx])
        encode_sentnece = tokenizer.encode(switch_sentence)
        
        diff = len(input_ids[i][sep_idx[1] + 1 : sep_idx[2]]) - len(torch.tensor(encode_sentnece)[1:-1])            
        input_ids[i][sep_idx[1] + 1 : sep_idx[2] - diff + 1] = torch.tensor(encode_sentnece)[1:]
        
    return input_ids


def train_model(model, loss_fn, optimizer, scaler, train_loader, test_loader, scheduler, config, tokenizer, save_path):
    prev_acc = 0
    for epoch in range(config.EPOCHS):
        epoch_acc = train_one_epoch(epoch, model, loss_fn, optimizer, scaler, train_loader, config.device, config.BATCH_SIZE, tokenizer, config, scheduler=scheduler)
        with torch.no_grad():
            test_acc = test_one_epoch(epoch, model, loss_fn, test_loader, config.device)
            if test_acc > prev_acc: 
                torch.save(model, save_path)
                prev_acc = test_acc
    return prev_acc


def make_submission(test_loader, models, submission_path, device):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
    output_pred = []

    with torch.no_grad():
        for i in range(len(models)):
            models[i].eval()
        for step, (input_ids, attention_mask, labels) in pbar:
            for i in range(len(models)):
                if i == 0:
                    y_preds = models[i](input_ids.to(device), attention_mask = attention_mask.to(device))[0]
                else:
                    y_preds += models[i](input_ids.to(device), attention_mask = attention_mask.to(device))[0]

            output_pred.append(int(torch.argmax(y_preds, 1)))

    output = pd.DataFrame(output_pred, columns=['pred'])
    output.to_csv(submission_path, index=False)


if __name__ == "__main__":
    config = get_config()
    print(config)
    seed_everything(2021)
    train_set, test_set, tokenizer = get_data(config)
    kfold = KFold(n_splits=5, shuffle=True)

    high_acc_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
        print(f"{fold} FOLD")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
        train_iter = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=train_subsampler)
        val_iter = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=val_subsampler)

        model, optimizer, scaler, scheduler, criterion = get_model(config)
        model.cuda()
        high_acc = train_model(model ,criterion, optimizer, scaler, train_iter, val_iter, scheduler, config, tokenizer, f"/opt/ml/model/{fold}_xlm-roberta-large_AdamW_focalloss0.5_mask_high.pt")
        high_acc_list.append(high_acc)
        #torch.save(model, f"/opt/ml/model/{fold}_xlm-roberta-large_AdamW_focalloss0.5_drop0.0_decay0.1_mask.pt")
    print(high_acc_list)

    models = []
    for i in range(5):
        models.append(torch.load(f"/opt/ml/model/{fold}_xlm-roberta-large_AdamW_focalloss0.5_mask_high.pt"))

    test_iter = DataLoader(test_set, batch_size = 1)
    make_submission(test_iter, models, "./submission.csv", config.device)
