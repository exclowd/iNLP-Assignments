from typing import Tuple
import torch
from torch import nn
import torchmetrics

from lightning import LightningModule
from elmo.models.components.rnn import RNNClassifier
from .elmo_module import ELMoLightningModule

class RNNClassifierLightningModule(LightningModule):
    def __init__(self,
                 net: RNNClassifier,
                 elmo_ckpt_path: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.elmo = ELMoLightningModule().load_from_checkpoint(
            checkpoint_path=elmo_ckpt_path,
        )
        self.elmo.freeze()

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.dev_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_loss = torchmetrics.MeanMetric()
        self.dev_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h1, h2 = self.elmo(x)
        x = torch.cat([c, h1, h2], dim=1)
        return self.net(x)


    def model_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor,
                                                  torch.Tensor,
                                                  torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=2)
        return loss, preds, y


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.log('dev_loss', loss)
        self.log('dev_acc', self.dev_acc(preds, batch[1]))
        self.dev_loss(loss)
        self.dev_acc(preds, targets)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(preds, batch[1]))
        self.test_loss(loss)
        self.test_acc(preds, targets)