import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# from Models.model import UpstreamTransformerXLSR
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()

        self.processor = Wav2Vec2Processor.from_pretrained("/root/autodl-tmp/facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained("/root/autodl-tmp/facebook/wav2vec2-xls-r-300m")

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.encoder.layers.parameters():
            param.requires_grad = False

        self.attn_pool = SelfAttentionPooling(HPARAMS['feature_dim'])

        self.language_classifier = nn.Sequential(
            nn.Linear(HPARAMS['feature_dim'], 512),
            nn.ReLU(),
            nn.Linear(512, 2))

        self.classification_criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = Accuracy(task="binary")
        self.f1_metric = F1Score(task="binary")
        self.lr = HPARAMS['lr']
        self.mixup_type = HPARAMS['mixup_type']

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def simple_forward(self, x, x_len):
        x = self.encoder(x).last_hidden_state
        x = self.attn_pool(x)
        language = self.language_classifier(x)
        return language

    def forward(self, x, x_len):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to(self.device)
        return self.simple_forward(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_l, x_len, filenames = batch
        # 确保y_l是张量类型
        if all(isinstance(item, int) for item in y_l):
            y_l = torch.tensor(y_l, dtype=torch.long, device=self.device)
        else:
            y_l = torch.stack(y_l)
        
        # 确保标签值在有效范围内（0和1，因为是二分类）
        y_l = torch.clamp(y_l, 0, 1)

        y_hat_l = self(x, x_len)        
        probs = F.softmax(y_hat_l, dim=1)

        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l)
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )
        train_step_acc = self.accuracy_metric(winners, y_l)
        train_step_f1 = self.f1_metric(winners, y_l)
        loss = language_loss

        # 在training_step中直接记录所有需要的epoch级别指标
        self.log("train/f1", train_step_acc, on_step=False, on_epoch=True)
        self.log("train/acc", train_step_f1, on_step=False, on_epoch=True)
        self.log("train/loss", language_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss':loss, 
                'probs': probs.detach().cpu().numpy(),
                'labels': y_l.detach().cpu().numpy().astype(int),
                'filenames': filenames,
                }
    

    def validation_step(self, batch, batch_idx):
        x, y_l, x_len, filenames  = batch
        # 确保y_l是张量类型
        if all(isinstance(item, int) for item in y_l):
            y_l = torch.tensor(y_l, dtype=torch.long, device=self.device)
        else:
            y_l = torch.stack(y_l)
        
        # 确保标签值在有效范围内（0和1，因为是二分类）
        y_l = torch.clamp(y_l, 0, 1)

        y_hat_l = self.simple_forward(x, x_len)
        language_loss = self.classification_criterion(y_hat_l, y_l)

        winners = y_hat_l.argmax(dim=1)
        corrects = (winners == y_l)
        language_acc = corrects.sum().float() / float( y_hat_l.size(0) )

        val_step_acc = self.accuracy_metric(winners, y_l)
        val_step_f1 = self.f1_metric(winners, y_l)

        # 在validation_step中直接记录所有需要的epoch级别指标
        self.log("val/acc", val_step_acc, on_step=False, on_epoch=True)
        self.log("val/loss", language_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", val_step_f1, on_step=False, on_epoch=True)  # 额外添加F1分数记录

        return {'val_loss':language_loss}
        # 注意：这里返回的字典内容可以简化，因为日志已经在step中记录
