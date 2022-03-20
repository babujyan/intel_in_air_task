import torch
import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):

    def __init__(self):
        super(ClassificationModel, self).__init__()
    
    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def test_epoch_end(self):
        pass

    def validation_epoch_end(self):
        pass