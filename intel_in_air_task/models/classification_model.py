import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torch.optim import SGD, lr_scheduler
from data.data_label_mapping import reverse_mapping


class ClassificationModel(pl.LightningModule):
    def __init__(self, unet, num_classes, optim_config):
        """
        Model constructor
        :param unet: backbone model
        :param num_classes: number of classes
        :param optim_config: optimizer configs
        """
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
        self.optim_config = optim_config

    def forward(self, input_img):
        """
        forward propagation
        :param input_img: image input
        :return: output mask
        """
        output = self.unet(input_img)
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

    def training_step(self, batch, batch_idx):
        image, label = batch
        label = label[:, 0]
        result = self.forward(image)
        loss = self.loss(result.float(), label.to(torch.int64))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        result = self.forward(image)
        label = label[:, 0]
        loss = self.loss(result.float(), label.to(torch.int64))
        preds = result.numpy().argmax()
        val_acc = np.mean(preds == label.numpy())
        n_true = np.sum(preds == label.numpy())
        return {"val_loss": loss, "val_acc": val_acc, "val_n_true": n_true}

    def test_step(self, batch, batch_idx, **kwargs):
        image, label = batch
        result = self.forward(image)
        loss = self.loss(label, result)
        preds = result.numpy().argmax()
        n_true = np.sum(preds == label.numpy())
        test_acc = np.mean(preds == label.numpy())
        return {"test_loss": loss, "test_acc": test_acc, "test_n_true": n_true}

    def test_epoch_end(self, outputs, dataloader_idx=0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        n_true = torch.stack([x['test_n_true'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': val_loss_mean,
                            "test_acc": n_true/len(outputs)}
        return {'test_loss': val_loss_mean, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs, dataloader_idx=0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        n_true = sum([x['val_n_true'] for x in outputs])
        tensorboard_logs = {'val_loss': val_loss_mean,
                            "val_acc": n_true/len(outputs)}
        return {'val_loss': val_loss_mean, "log": tensorboard_logs}

    def predict_step(self, batch, batch_idx, **kwargs):
        self.eval()
        with torch.no_grad():
            batch = batch.unsqueeze(dim=0)
            result = self.forward(batch)
            preds = result.numpy().argmax()

        return {f"Prediction_{batch_idx}": reverse_mapping[preds]}

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.optim_config['lr'])
        schedule = lr_scheduler.ExponentialLR(optimizer,
                                              **self.optim_config['scheduler'])
        schedule_dict = {
            'scheduler': schedule,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'loss',
        }
        return [optimizer], [schedule_dict]
