
import random
import glob
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl

class BertForSequenceClassificationMultiLabel(torch.nn.Module):

  def __init__(self, model_name, num_labels):
    super().__init__()
    self.bert = BertModel.from_pretrained(model_name)
    self.linear = torch.nn.Linear(
        self.bert.config.hidden_size, num_labels
    )

  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
    
    #最終層
    bert_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    last_hidden_state = bert_output.last_hidden_state

    #平均
    averaged_hidden_state = \
      (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
      / attention_mask.sum(1, keepdim=True)

    #線形変換
    scores = self.linear(averaged_hidden_state)

    #出力形式
    output = {'logits':scores}

    #ラベルが入力に含まれていたら損出計算
    if labels is not None:
      loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float())
      output['loss'] = loss

    #属性でアクセスできるようにする
    output = type('bert_output', (object,), output) 
    
    return output