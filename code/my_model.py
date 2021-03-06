import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

class Mymodel(nn.Module):
    def __init__(self, model_name):
        super().__init__() 
        self.transformer_model = AutoModel.from_pretrained(model_name, hidden_dropout_prob=0.0)
        self.lstm = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.3, bidirectional = True, batch_first = True)
        self.dense_layer = nn.Linear(2048, 42, bias=True)
        
    
    def forward(self, input_ids, attention_mask):
        encode_layers = self.transformer_model(input_ids=input_ids, attention_mask = attention_mask)[0]
        enc_hiddens, (last_hidden, last_cell) = self.lstm(encode_layers)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)

        output = [self.dense_layer(output_hidden)]

        return output