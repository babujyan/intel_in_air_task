import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.nn.functional import one_hot

class ClassificationModel(pl.LightningModule):

    def __init__(self, unet, num_classes):
        super(ClassificationModel, self).__init__()
        self.unet = unet
        self.conv1 = nn.Conv2d(1024, 500, 5)
        self.conv2 = nn.Conv2d(500, 100, 5)
        self.conv3 = nn.Conv2d(100, 1, 1)
        self.bn1 = nn.BatchNorm2d(500)
        self.bn2 = nn.BatchNorm2d(100)
        self.act = nn.ReLU()
        self.loss = nn.NLLLoss()
        self.last_fc = nn.Linear(64, num_classes)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.num_classes = num_classes
    
    def forward(self, input):
        output = self.unet(input)
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.act(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act(output)
        output = self.conv3(output)
        output = self.flatten(output)
        output = self.last_fc(output)
        output = self.softmax(output)
        return output

    def training_step(self, batch):
        image, label = batch
        result = self.forward(image)
        print(one_hot(label, num_classes=self.num_classes), result)
        loss = self.loss(one_hot(label, num_classes=self.num_classes), result)
        return {"loss": loss}

    def validation_step(self, batch):
        image, l = batch
        result = self.forward(image)
        loss = self.loss(l, result)
        preds = result.numpy().argmax()
        val_acc = np.mean(preds == l.numpy())
        n_true = np.sum(preds == l.numpy())
        return {"val_loss": loss, "val_acc": val_acc, "val_n_true": n_true}

    def test_step(self, batch):
        image, l = batch
        result = self.forward(image)
        loss = self.loss(l, result)
        preds = result.numpy().argmax()
        n_true = np.sum(preds == l.numpy())
        test_acc = np.mean(preds == l.numpy())
        return {"test_loss": loss, "test_acc": test_acc, "test_n_true": n_true}

    def test_epoch_end(self, outputs, dataloader_idx=0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        n_true = torch.stack([x['test_n_true'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean, "test_acc": n_true/len(outputs)}
        return {'test_loss': val_loss_mean, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        n_true = torch.stack([x['val_n_true'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss': val_loss_mean, "val_acc": n_true/len(outputs)}
        return {'val_loss': val_loss_mean, "log": tensorboard_logs}

    def configure_optimizers(self):
        return super().configure_optimizers()