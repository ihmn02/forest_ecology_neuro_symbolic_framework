import comet
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from os.path import join

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from test import canopy_test
from paths import data_path


class FrickerModel(pl.LightningModule):
    def __init__(self, num_classes, img_depth, init_num_filt=32, max_num_filt=128, num_layers=7, lr=1e-4, rule_ind=5, class_wts=None):
        super(FrickerModel, self).__init__()
        self.convs = []
        num_filt = init_num_filt
        out_filt = 2 * num_filt
        self.convs.append(nn.Conv2d(img_depth, num_filt, 3))
        self.convs.append(nn.ReLU())
        for i in range(num_layers):
            self.convs.append(nn.Conv2d(num_filt, out_filt, 3))
            num_filt = out_filt
            if out_filt < max_num_filt:
                out_filt = 2 * num_filt
            self.convs.append(nn.ReLU())
        self.convs.append(nn.Conv2d(out_filt, num_classes, 1))
        self.net = nn.Sequential(*self.convs)
        self.lr = lr
        self.class_wts = class_wts
        self.rule_ind = rule_ind

        print("Passed values check:\n")
        print("  lr: {}\n".format(self.lr))
        print("  wts: {}".format(self.class_wts))
        print()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        self.train()
        batch_train_x, batch_train_chm, batch_train_dem, batch_train_y = batch
        output = self(batch_train_x).reshape(-1, 8)

        temp_cwts = None if self.class_wts is None else self.class_wts.to(self.device)
        loss_task_func = nn.CrossEntropyLoss(weight=temp_cwts)
        loss_task = loss_task_func(output, batch_train_y)

        loss = loss_task

        # log to comet
        with comet.experiment.train():
            comet.experiment.log_metric("train_loss", loss, epoch=self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        val_x, val_chm, val_dem, val_y = batch
        output = self(val_x).reshape(-1, 8)

        temp_cwts = None if self.class_wts is None else self.class_wts.to(self.device)
        loss_task_func = nn.CrossEntropyLoss(weight=temp_cwts)
        val_loss_task = loss_task_func(output, val_y).item()

        y_true = val_y.cpu().numpy()
        y_score = output.cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        val_acc = 100 * accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        print(confusion_matrix(y_true, y_pred))
        print("-----------------------------------------------------")
        print(classification_report(y_true, y_pred))
        self.log("val_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", val_loss_task, prog_bar=True, on_step=False, on_epoch=True)

        # log data to comet
        with comet.experiment.validate():
           comet.experiment.log_metric("val_loss", val_loss_task, epoch=self.current_epoch)
           comet.experiment.log_metric("val_acc", val_acc, epoch=self.current_epoch)
           comet.experiment.log_metric("val_f1", val_f1, epoch=self.current_epoch)

        # log the confusion matrix to comet
        comet.experiment.log_confusion_matrix(y_true, y_pred, title="val con. mat. epoch: " + str(self.current_epoch))
        return {'val_loss':val_loss_task, 'val_acc':val_acc, "val_f1":val_f1}

    def test_step(self, batch, batch_idx):
        te_x, te_chm, te_dem, te_y = batch
        self.eval()
        #canopy_test("hsi", self)
        output = self(te_x).reshape(-1, 8)

        temp_cwts = None if self.class_wts is None else self.class_wts.to(self.device)
        loss_task_func = nn.CrossEntropyLoss(weight=temp_cwts)
        test_loss_task = loss_task_func(output, te_y).item()

        y_true = te_y.cpu().numpy()
        y_score = output.cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        test_f1 = f1_score(y_true, y_pred, average="macro")
        test_acc = 100 * accuracy_score(y_true, y_pred)
        class_rep = classification_report(y_true, y_pred)
        class_rep_dict = classification_report(y_true, y_pred, output_dict=True)
        self.log("class_f1", class_rep_dict[str(self.rule_ind)]['f1-score'], prog_bar=False, on_step=False, on_epoch=True)
        self.log("class_prec", class_rep_dict[str(self.rule_ind)]['precision'], prog_bar=False, on_step=False, on_epoch=True)
        self.log("class_rec", class_rep_dict[str(self.rule_ind)]['recall'], prog_bar=False, on_step=False, on_epoch=True)
        self.log("class_supp", class_rep_dict[str(self.rule_ind)]['support'], prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_loss", test_loss_task, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", test_f1, prog_bar=True, on_step=False, on_epoch=True)
        print("\n\n")
        print("test_acc: ", test_acc)
        print("test_f1: ", test_f1)
        print()
        print(confusion_matrix(y_true, y_pred))
        print("-----------------------------------------------------")
        print(class_rep)        

        # log the confusion matrix to comet
        comet.experiment.log_confusion_matrix(y_true, y_pred, title="test confusion matrix")        
  
        # store the predictions and labels in a csv for analysis
        self.store_preds(y_true, y_pred, te_x, te_chm, te_dem)

        return {'test_loss':test_loss_task, 'test_acc':test_acc, 'test_f1':test_f1}

    def store_preds(self, y_true, y_pred, te_x, te_chm, te_dem):
        max_ht= torch.amax(te_chm, dim=(2, 3)).cpu().numpy().flatten()
        max_elev = torch.amax(te_dem, dim=(2,3 )).cpu().numpy().flatten()
        patch_sum = torch.sum(te_x[:, :, 5:8, 5:8], (2, 3))
        gli = ((2.0 * patch_sum[:, 1]) - patch_sum[:, 0] - patch_sum[:, 2]) / ((2.0 * patch_sum[:, 1]) + patch_sum[:, 0] + patch_sum[:, 2])
        gli = gli.cpu().numpy().flatten()
        df = pd.DataFrame({"ground_truth":y_true, "preds":y_pred, "max_ht":max_ht, "max_elev":max_elev, "gli":gli})
        fold_num = int(data_path[-1])  
        fname = "preds_base_fold-" + str(fold_num) + "_expid-"+ comet.experiment.id + ".csv"
        df.to_csv(join(data_path, fname), header=True, index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
