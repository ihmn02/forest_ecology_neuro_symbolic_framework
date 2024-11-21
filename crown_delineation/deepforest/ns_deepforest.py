# entry point for deepforest model
import importlib
import os
import typing
import warnings

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio

import torch
from torch import nn
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import optim
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

from deepforest import dataset, visualize, get_data, utilities, predict
from deepforest import evaluate as evaluate_iou

from deepforest.main import deepforest 

from rules import Rule1, Rule2

class Area_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        logits = self.fc_net(x)
        return logits

class Ns_deepforest(deepforest):
    def __init__(
                 self,
         num_classes: int = 1,
         label_dict: dict = {"Tree": 0},
         transforms=None,
         config_file: str = 'deepforest_config.yml',
         config_args=None,
         model=None,
         existing_train_dataloader=None,
         existing_val_dataloader=None,
         scale=1.,
         rule_lambdas=[1.]
    ):
        super().__init__(
             num_classes,
             label_dict,
             transforms,
             config_file,
             config_args,
             model,
             existing_train_dataloader,
             existing_val_dataloader
        )
        #self.area_net = Area_net()
        self.rule_lambdas = rule_lambdas
        self.scale = scale
        self.pi_params = [0.995, 0.7]
        self.batch_cnt = 0
        self.rules = [
            Rule1("r1", "all(x) bbox_area_lt_site_mean(x) <==> tree(x)", self.rule_lambdas[0]),
            Rule2("r2", "all(x) bbox_area_lte_hca_area_prediction(x) <==> tree(x)", self.rule_lambdas[1])            
        ]

        # freeze retinanet weights
        #for param in self.model.parameters():
        #    param.requires_grad = False


    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Confirm model is in train mode
        self.model.train()

        bce_loss = torch.nn.BCEWithLogitsLoss()                   #  torch.nn.BCELoss()  
        self.batch_cnt += 1

        # allow for empty data if data augmentation is generated
        path, images, chms, targets = batch
        train_loss_dict_task = self.model.forward(images, targets)

        # put model in eval mode
        self.model.eval()

        # preds a list of dictionaries
        # one dictionary per image
        # each dictionary has keys 'boxes', 'scores', and 'labels'
        preds = self.model.forward(images)          #targets must be included in training mode

        # get special features
        bbox_coords = torch.cat([pred['boxes'] for pred in preds], dim=0)
        bbox_areas = self.bbox_area(bbox_coords).reshape(-1, 1)
        preds = self.get_heights(chms, preds)
        bbox_max_hts = torch.cat([pred['hts'] for pred in preds], dim=0).reshape(-1, 1)
        #preds_area = self.area_net(bbox_areas).reshape(-1, 1)
        
        #rule_fxn_outputs = [preds_area]

        curr_iter = self.batch_cnt / 34
        pi = self.get_pi(curr_iter, 5)

        # task losses
        #train_loss_task_area = bce_loss(preds_area, torch.cat([pred['scores'] for pred in preds], dim=0).reshape(-1, 1))
        #train_loss_dict_task['classification_area'] = pi * train_loss_task_area

        # rule losses
        train_tot_loss_rule, train_ver_ratio = self.process_rules(images, chms, preds, bbox_areas, bbox_max_hts, bce_loss)

        # sum of regression and classification loss
        train_tot_task_loss = sum([loss for loss in train_loss_dict_task.values()])

        losses = (pi * self.scale * train_tot_loss_rule) + (1-pi) * (train_tot_task_loss)

        self.log('tot_loss', losses, prog_bar=True, on_step=True) 
        self.log('task_loss', train_tot_task_loss, prog_bar=True, on_step=True)
        self.log('rule_loss', train_tot_loss_rule, prog_bar=True, on_step=True)
        self.log('pi', pi, prog_bar=True, on_step=True)
        
        return losses

    def validation_step(self, batch, batch_idx):
        """Evaluate a batch
        """
        try:
            path, images, chms, targets = batch
        except:
            print("Empty batch encountered, skipping")
            return None

        # Get loss from "train" mode, but don't allow optimization
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        self.model.eval()
        preds = self.model.forward(images)

        # Calculate intersection-over-union
        self.iou_metric.update(preds, targets)
        self.mAP_metric.update(preds, targets)

        # Log loss
        for key, value in loss_dict.items():
            self.log("val_{}".format(key), value, on_epoch=True)

        for index, result in enumerate(preds):
            boxes = visualize.format_boxes(result)
            boxes["image_path"] = path[index]
            self.predictions.append(boxes)

        return losses

    def load_dataset(self,
                     csv_file,
                     root_dir=None,
                     augment=False,
                     shuffle=True,
                     batch_size=1,
                     train=False):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            augment: Whether to create a training dataset, this activates data augmentations
            
        Returns:
            ds: a pytorch dataset
        """
        ds = dataset.NsTreeDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 transforms=self.transforms(augment=augment, ns_trans=True),
                                 label_dict=self.label_dict,
                                 preload_images=self.config["train"]["preload_images"])

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=utilities.collate_fn,
            num_workers=self.config["workers"],
        )

        return data_loader


    def predict_file(self, csv_file, root_dir, savedir=None, color=None, thickness=1):
        """Create a dataset and predict entire annotation file
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            savedir: Optional. Directory to save image plots.
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """
        df = pd.read_csv(csv_file)
        ds = dataset.NsTreeDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 transforms=None,
                                 train=False)
        dataloader = self.predict_dataloader(ds)

        results = predict._dataloader_wrapper_(model=self,
                                               trainer=self.trainer,
                                               annotations=df,
                                               dataloader=dataloader,
                                               root_dir=root_dir,
                                               nms_thresh=self.config["nms_thresh"],
                                               savedir=savedir,
                                               color=color,
                                               thickness=thickness)

        return results


    def get_pi(self, cur_iter, pi_s, pi=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        alpha, pi_0 = self.pi_params[0], self.pi_params[1]
        pi = 1. - max([alpha ** cur_iter, pi_0])
        if (self.current_epoch < pi_s):
            pi = 0
        return pi


    def bbox_area(self, pred_bboxes):
        """
        calculates the area of each predicted bounding boxes
        """

        x_len = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        y_len = pred_bboxes[:, 3] - pred_bboxes[:, 1]
        bb_area = x_len * y_len
        bb_area = bb_area.reshape(-1, 1)
        
        # expand the area feature from a scalar to 4-vector and normalize
        #bb_area = torch.cat([torch.ones(bb_area.shape).to(self.device), bb_area, torch.square(bb_area), bb_area ** 3], dim=1)
        #bb_area = torch.nn.functional.normalize(bb_area)
        return bb_area


    def process_rules(self, x, x_chm, output, bb_areas, bb_max_hts, loss_fxn):
        rule_loss = []
        ver_ratio = []
        for rnum, rule in enumerate(self.rules):
           # rule fxn 
           rule_fxn_output = rule.rule_fxn(x, chm=x_chm, bb_areas=bb_areas, bb_max_hts=bb_max_hts)     # for manual rule fxn

           rule.generic_interface(torch.cat([preds['scores'] for preds in output], dim=0).reshape(-1, 1), rule_fxn_output)
           r_out = rule.get_val()
           loss_r = 1.0 - r_out.mean()              #loss_fxn(r_out, torch.ones(r_out.shape, dtype=torch.float).to(r_out.device))

           loss_r = rule.lmbda * loss_r
           rule_loss.append(loss_r)
           num_true_rules = torch.sum(r_out >= 0.9)
           ver_ratio.append((num_true_rules/r_out.shape[0]).item())         # number of true rules/tot. number of rules 
    
        tot_loss_rule = torch.sum(torch.stack(rule_loss))
        return (tot_loss_rule, ver_ratio)



    def get_heights(self, chms, preds): 
        for idx in range(len(preds)):
            ht_list = []
            chm = chms[idx]
            boxes = preds[idx]['boxes'].clone().detach()
            boxes = torch.round(boxes) 
            boxes = boxes.int()

            for row in range(boxes.shape[0]):
                if torch.numel(chm) > 0:
                   # +1 eliminates degenrate boxes i.e. boxes with 0 width or height
                   ht = torch.amax(chm[0, boxes[row][1]:boxes[row][3]+1, boxes[row][0]:boxes[row][2]+1], dim=(0, 1))
                else:
                   ht = 0.
                ht_list.append(ht)

            hts = torch.vstack(ht_list)
            preds[idx]['hts'] = hts


        return preds
