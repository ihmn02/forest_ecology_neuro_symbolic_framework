import comet

import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from test import canopy_test
from paths import data_path
from utils_learning import verification

from rules import Rule1, Rule2, Rule3

class Network(nn.Module):
    def __init__(self, num_classes, img_depth, init_num_filt=32, max_num_filt=128, num_layers=6):
        super(Network, self).__init__()
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

    def forward(self, x):
        return self.net(x)


class NsFrickerModel(pl.LightningModule):
    def __init__(self, num_classes, img_depth, lambdas, scale, thresholds, init_num_filt=32, max_num_filt=128, num_layers=7, lr=1e-4, class_wts=None):
        super(NsFrickerModel, self).__init__()
        self.num_classes = num_classes
        self.img_depth = img_depth
        self.lambdas = lambdas
        self.scale = scale
        self.thresholds = thresholds
        self.init_num_filt = init_num_filt
        self.max_num_filt = max_num_filt
        self.num_layers = num_layers
        self.lr = lr
        self.class_wts = class_wts 

        self.rules = [
                        Rule1("r1", "all(x) chm_gt_46m(x) ==> ~black_oak(x)", self.lambdas[0]),
                        Rule2("r2", "all(x) chm_gt_53m(x) ==> ~lodgepole_pine(x)", self.lambdas[1]),
                        #Rule3("r3", "all(x) dem_gt_2072m(x) ==> ~red_fir(x)", self.lambdas[2])
        ]
        
        # set rule thresholds
        for idx, r in enumerate(self.rules):
            r.set_thr(thresholds[idx])

        self.r_loss = []
        self.t_loss = []
        self.tot_loss = []

        print("Passed values check:\n")
        print("  thr: {}\n  scale: {}\n  lr: {}\n".format(self.thresholds, self.scale, self.lr))
        print("  lambdas: {}\n".format(str([round(x, 3) for x in self.lambdas])))
        print("  wts: {}".format(self.class_wts))
        print()

        self.net_task = Network(self.num_classes, self.img_depth-1, init_num_filt=self.init_num_filt, max_num_filt=self.max_num_filt, num_layers=self.num_layers)


    def forward(self, x, x_chm, x_dem):
        return self.net_task(x)

        

    def training_step(self, batch, batch_idx):
        temp_class_wts = None if self.class_wts	is None	else self.class_wts.to(self.device)

        ce_loss = torch.nn.CrossEntropyLoss(weight=temp_class_wts) 
  
        self.train()
        train_x, train_chm, train_dem, train_y = batch
        output = self(train_x[:, :3, :, :], train_chm, train_dem)   # only take rgb channels; no aux data
        output = output.reshape(-1, 8)

        rule_fxn_outputs = [self.rules[0].rule_fxn(train_x[:, -1, :, :]), self.rules[1].rule_fxn(train_x[:, -1, :, :])]
   
        y_score = output.clone().detach()
        _, y_pred = torch.max(y_score, dim=1)  #torch.argmax(y_score, dim=1)


        max_ht = torch.amax(train_x[:, -1, :, :], dim=(1, 2))        # only used for testing; max height taken inside rule_fxn
        loss_task1 = ce_loss(output, train_y)
        task_losses = [loss_task1]
        tot_loss_task = torch.sum(torch.stack(task_losses))

        # execute each rule on batch
        rule_loss = []
        for rnum, rule in enumerate(self.rules):
           # rule fxn                                                                                                             
           #rule_fxn_output = rule.rule_fxn(train_x[:, -1, :, :], chm=train_chm, dem=train_dem)

           rule.generic_interface(output, rule_fxn_outputs[rnum])
           r_out = rule.get_val()
           loss_r = 1.0 - r_out.mean()        #bce_loss(r_out, torch.ones(r_out.shape, dtype=torch.float).to(r_out.device))
                            
           loss_r = rule.lmbda * loss_r
           rule_loss.append(loss_r) 

        tot_loss_rule = torch.sum(torch.stack(rule_loss))

        loss = (self.scale * tot_loss_rule) + tot_loss_task

        self.log("task_loss", tot_loss_task, prog_bar=True, on_step=True, on_epoch=False)
        self.log("rule_loss", tot_loss_rule, prog_bar=True, on_step=True, on_epoch=False)
        self.r_loss.append(tot_loss_rule.item())
        self.t_loss.append(tot_loss_task.item())
        self.tot_loss.append(loss.item())

        # log to comet
        #with comet.experiment.train():
        #   comet.experiment.log_metric("loss_task", loss_task, epoch=self.current_epoch)
        #   comet.experiment.log_metric("loss_rule", self.scale * tot_loss_rule, epoch=self.current_epoch)
        #   comet.experiment.log_metric("tot_loss", loss, epoch=self.current_epoch)
            
        return loss

    def training_epoch_end1(self, training_step_outputs):
        rule_loss = np.array(self.r_loss)
        task_loss = np.array(self.t_loss)        

        #rule_loss = rule_loss[rule_loss > 0]
        rule_mean = np.mean(rule_loss)
       
        task_mean = np.mean(task_loss)

        print("mean_task_loss: {} \t mean_rule_loss: {}".format(task_mean, rule_mean))

    def validation_step(self, batch, batch_idx):
        self.eval()

        temp_class_wts = None if self.class_wts	is None	else self.class_wts.to(self.device)
        ce_loss = torch.nn.CrossEntropyLoss(weight=temp_class_wts)
        bce_loss =  torch.nn.BCEWithLogitsLoss()        #torch.nn.BCELoss()             
        class_f1 = []

        val_x, val_chm, val_dem, val_y = batch  

        output = self(val_x[:, :3, :, :], val_chm, val_dem)
        output = output.reshape(-1, 8)
        
        rule_fxn_outputs = [self.rules[0].rule_fxn(val_x[:, -1, :, :]), self.rules[0].rule_fxn(val_x[:, -1, :, :])]

        y_true = val_y.cpu().numpy()
        y_score = output.cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)

        val_acc = 100 * accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro')

        class_rep_dict = classification_report(y_true, y_pred, output_dict=True)
        class_f1 = [class_rep_dict[str(r.rule_ind)]['f1-score'] for r in self.rules]

        val_loss_task1 = ce_loss(output, val_y)

        val_task_losses = [val_loss_task1]
        val_tot_loss_task = torch.sum(torch.stack(val_task_losses))

        val_tot_loss_rule, val_ver_ratio = self.get_ver_rat(val_x, val_chm, val_dem, output, rule_fxn_outputs)

        val_loss = (self.scale * val_tot_loss_rule) + val_tot_loss_task 

        print()
        print("-----------------------------------------------------")
        print()
        print(confusion_matrix(y_true, y_pred))
        print()
        print("-----------------------------------------------------")
        print()
        print(classification_report(y_true, y_pred))
        print()
        print("-----------------------------------------------------")
        for idx, r in enumerate(val_ver_ratio):
            print(f"\nrule_{idx+1} v. ratio = {r}\n")
        print("-----------------------------------------------------")

        self.log("val_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        # log data to comet
        with comet.experiment.validate():
           #comet.experiment.log_metric("val_acc", val_acc, epoch=self.current_epoch)
           #comet.experiment.log_metric("val_f1", val_f1, epoch=self.current_epoch)
           #comet.experiment.log_metric("val_loss", val_loss, epoch=self.current_epoch)
           for idx, val in enumerate(val_ver_ratio):
               comet.experiment.log_metric(f"ver_rat_r{idx+1}", val, epoch=self.current_epoch)
           for idx, val in enumerate(class_f1):
               comet.experiment.log_metric(f"class_f1_r{idx+1}", val, epoch=self.current_epoch)
        

        # log the confusion matrix to comet
        comet.experiment.log_confusion_matrix(y_true, y_pred, title="val con. mat. epoch: " + str(self.current_epoch))
        
        return {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_ver_rat':val_ver_ratio}

    def test_step(self, batch, batch_idx):
        self.eval(
)
        temp_class_wts = None if self.class_wts	is None	else self.class_wts.to(self.device)
        ce_loss = torch.nn.CrossEntropyLoss(weight=temp_class_wts)

        te_x, te_chm, te_dem, te_y = batch                     
    
        # TO DO modify model in canopy_test() to use both x and chm
        #canopy_test("hsi", self)
        output = self(te_x[:, :3, :, :], te_chm, te_dem)
        output = output.reshape(-1, 8)

        rule_fxn_outputs = [self.rules[0].rule_fxn(te_x[:, -1, :, :]), self.rules[1].rule_fxn(te_x[:, -1, :, :])]

        test_tot_loss_rule, test_ver_ratio = self.get_ver_rat(te_x, te_chm, te_dem, output, rule_fxn_outputs)

        test_loss_task1 = ce_loss(output, te_y)

        test_task_losses = [test_loss_task1]
        test_tot_loss_task = torch.sum(torch.stack(test_task_losses))


        test_loss = (self.scale * test_tot_loss_rule) + test_tot_loss_task

        y_true = te_y.cpu().numpy()
        y_score = output.cpu().numpy()
        y_pred = np.argmax(y_score, axis=1)
        test_acc = 100 * accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='macro')
        class_rep = classification_report(y_true, y_pred)
        class_rep_dict = classification_report(y_true, y_pred, output_dict=True)
        class_f1 = [class_rep_dict[str(r.rule_ind)]['f1-score'] for r in self.rules]

        self.log("test_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", test_f1, prog_bar=True, on_step=False, on_epoch=True)
        print("test_f1", test_f1)
        print("test_acc: ", test_acc)
        print()
        print("-----------------------------------------------------")
        print()
        print(confusion_matrix(y_true, y_pred))
        print()
        print("-----------------------------------------------------")
        print()
        print(class_rep)
        print()
        print("-----------------------------------------------------")
        for idx, r in enumerate(test_ver_ratio):
            print(f"\nrule_{idx+1} v. ratio = {r}\n")
        print("-----------------------------------------------------")
       
        # log data to comet
        with comet.experiment.test():
           #comet.experiment.log_metric("test_acc", test_acc, epoch=self.current_epoch)
           #comet.experiment.log_metric("test_f1", test_f1, epoch=self.current_epoch)

           for idx, val in enumerate(test_ver_ratio):
               comet.experiment.log_metric(f"ver_rat_r{idx+1}", val)

           for idx, val in enumerate(class_f1):
               comet.experiment.log_metric(f"class_f1_r{idx+1}", val)

        # log the confusion matrix to comet
        comet.experiment.log_confusion_matrix(y_true, y_pred, title="test confusion matrix")

        # store the predictions and labels in a csv for analysis
        #self.store_preds(y_true, y_pred, te_x, te_chm, te_dem)


        res_dict = {
                       'test_loss': test_loss, 
                       'test_acc': test_acc,
                       'test_f1': test_f1, 
                       'test_ver_ratio': test_ver_ratio,
                       'class_f1': [class_rep_dict[str(r.rule_ind)]['f1-score'] for r in self.rules], 
                       'class_prec': [class_rep_dict[str(r.rule_ind)]['precision'] for r in self.rules],
                       'class_rec': [class_rep_dict[str(r.rule_ind)]['recall'] for r in self.rules], 
                       'class_sup': [class_rep_dict[str(r.rule_ind)]['support'] for r in self.rules],
                       'rule_ind': [r.rule_ind for r in self.rules],
                       'lambdas': [r.lmbda for r in self.rules],
                       'thr': [r.thr for r in self.rules],
                       'r_names': [r.rule_name for r in self.rules]
        }
        self.results_dict = res_dict
        return res_dict

    def store_preds(self, y_true, y_pred, te_x, te_chm, te_dem):
        max_ht= torch.amax(te_chm, dim=(2, 3)).cpu().numpy().flatten()
        max_elev = torch.amax(te_dem, dim=(2,3 )).cpu().numpy().flatten()
        patch_sum = torch.sum(te_x[:, :, 5:8, 5:8], (2, 3))
        gli = ((2.0 * patch_sum[:, 1]) - patch_sum[:, 0] - patch_sum[:, 2]) / ((2.0 * patch_sum[:, 1]) + patch_sum[:, 0] + patch_sum[:, 2])
        gli = gli.cpu().numpy().flatten()
        df = pd.DataFrame({"ground_truth":y_true, "preds":y_pred, "max_ht":max_ht, "max_elev":max_elev, "gli":gli})
        fold_num = int(data_path[-1])
        fname = "preds_nsf_fold-" + str(fold_num) + "_expid-"+ comet.experiment.id + ".csv"
        df.to_csv(join(data_path, fname), header=True, index=False)    # TODO restore this line

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


    def get_pi(self, cur_iter, pi=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        k, lb = self.pi_params[0], self.pi_params[1]
        pi = 1. - max([k ** cur_iter, lb])
        return pi

    def get_ver_rat(self, x, x_chm, x_dem, output, rule_fxn_outputs):
        rule_loss = []
        ver_ratio = []
        for rnum, rule in enumerate(self.rules):
           # rule fxn                                                                                                             
           #rule_fxn_output = rule.rule_fxn(x[:, -1, :, :], chm=x_chm, dem=x_dem)

           rule.generic_interface(output, rule_fxn_outputs[rnum])
           r_out = rule.get_val()
           loss_r = 1.0 - r_out.mean()              #loss_fxn(r_out, torch.ones(r_out.shape, dtype=torch.float).to(r_out.device))

           loss_r = rule.lmbda * loss_r
           rule_loss.append(loss_r)
           num_true_rules = torch.sum(r_out >= 0.9)
           ver_ratio.append((num_true_rules/r_out.shape[0]).item())         # number of true rules/tot. number of rules 
    
        tot_loss_rule = torch.sum(torch.stack(rule_loss))
        return (tot_loss_rule, ver_ratio)
