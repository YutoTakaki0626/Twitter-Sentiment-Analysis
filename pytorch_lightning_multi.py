
import random
import glob
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl

from multi_classification import BertForSequenceClassificationMultiLabel

class BertForSequenceClassificationMultiLabel_pl(pl.LightningModule):

  def __init__(self, model_name, num_labels, lr):
    super().__init__()
    self.save_hyperparameters()
    self.bert_scml = BertForSequenceClassificationMultiLabel(
        model_name, num_labels=num_labels
    )

  def training_step(self, batch, batch_idx):
    output = self.bert_scml(**batch)
    loss = output.loss
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    output = self.bert_scml(**batch)
    val_loss = output.loss
    self.log('val_loss', val_loss)

  def test_step(self, batch, batch_idx):
    labels = batch.pop('labels')
    output = self.bert_scml(**batch)
    scores = output.logits
    labels_predicted = (scores > 0).int()
    num_correct = ( labels_predicted == labels ).all(-1).sum().item()
    accuracy = num_correct / scores.size(0)
    self.log('accuracy', accuracy)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)