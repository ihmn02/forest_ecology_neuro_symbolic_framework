# TO DO remove warning supression
import comet

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from os.path import join

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.distributions.beta import Beta
from torchsummary import summary
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import Hsi_raster, trans
from model import Net, Frickernet
#from test import canopy_test
from frickernet_pl import FrickerModel
from ns_frickernet_pl import NsFrickerModel
from hyperparameter_tune import run_ax
from paths import train_data_uri, test_data_uri, val_data_uri, data_path
from pars import args
import my_log as mlg

def main():
    device = 'gpu'
    seed = args.seed # 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = args.batch
    n_trials = args.n_trials
    

    train_ds = Hsi_raster(data_path, train_data_uri, transform=None, test=False, aux_data=args.aux_data)

    val_set_size = round(0.1 * len(train_ds) )
    train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_set_size, val_set_size], generator=torch.Generator().manual_seed(42))
    #train_ds, val_ds = random_split(train_ds, [len(train_ds) - 12004, 12004])
    #train_ds, val_ds, _  = random_split(train_ds, [1024, 512, len(train_ds)-1024-512], generator=torch.Generator().manual_seed(42))

    #val_ds = Hsi_raster(data_path, val_data_uri, transform=None, test=True, aux_data=args.aux_data)
    test_ds = Hsi_raster(data_path, test_data_uri, transform=None, test=True, aux_data=args.aux_data)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0)
    print("\nbaseline model?: {}".format(args.baseline))
    print("\nhyperparameter tune:? {}".format(args.htune))
    print("\ndata size: {}/{}/{}".format(len(train_ds), len(val_ds), len(test_ds)))
    print("\nbatch size: {}".format(args.batch))
    print("\nimage depth : {}".format(train_ds[0][0].shape[0]))
    print("\naux data: {}".format(args.aux_data)) 
    print("\nseed: {}".format(seed))
    print("\nn_trials: {}".format(args.n_trials))

    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(8), y=train_ds.dataset.y)
    print('\nclass weights: ', class_weights)
    class_weight_dict = {}
    for i in range(8):
        class_weight_dict[i] = class_weights[i]
    class_weight_tensor = torch.tensor(list(class_weight_dict.values()), dtype=torch.float32)
    #class_weight_tensor = None 

    n_layers = 6
    output_dim = 8
    init_num_filt=32
    max_num_filt=128
    img_depth= train_ds[0][0].shape[0]
    lr = args.lr
    scale = args.scale
    thr = args.thr
    lambdas = args.lambdas
    epochs = args.epochs
    wdir = args.wdir
    aux_data = args.aux_data

    # baseline model has none of these parameters
    if args.baseline:
        scale = "N/A"          

    print('\nscale:{}\tlr: {}\n'.format(scale, lr))
    print('\nthreshold: {}\n'.format(thr))
    print('\nlambdas: {}\n'.format(lambdas))                     

    if args.htune:
        run_ax(device, train_dataloader, val_dataloader, test_dataloader, n_layers, output_dim, init_num_filt, max_num_filt, img_depth, lambdas, thr, scale, lr, epochs, n_trials, seed, class_weight_tensor)
    else:
        if args.baseline:
            baseline_model = FrickerModel(output_dim, img_depth, init_num_filt, max_num_filt, n_layers, lr, rule_ind=5, class_wts=class_weight_tensor)
            checkpoint_callback = ModelCheckpoint(monitor='val_f1', 
                                                  auto_insert_metric_name=False, 
                                                  save_top_k=1, 
                                                  mode='max',
                                                  every_n_epochs=1,
                                                  filename="baseline-epoch{epoch:02d}-val-f1_{val_f1:.2f}")
            trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], accelerator=device, devices=-1)
            trainer.fit(baseline_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            results = trainer.test(baseline_model, dataloaders=test_dataloader, ckpt_path="best")
            #if args.log_file is not None:
            #   csv_wr_obj = mlg.Writer(args.log_file, ["tag", "exp_id", "seed", "batch_size", "lr", "epochs", "aux_data", "wdir",
            #                                           "test_f1", "test_acc", "test_loss", "class_f1", "class_prec", "class_rec", "ver_ratio", "class_support"])

            #   # log can only store scalar values; have to store in model for arrays
            #   for r_num, r_ind in enumerate(trainer.model.results_dict['rule_ind']):
            #       csv_wr_obj.write_data([
            #                                args.tag,
            #                                comet.experiment.id,
            #                                trainer.model.results_dict['r_names'][r_num],
            #                                seed,
            #                                batch_size,
            #                                lr,
            #                                epochs,
            #                                aux_data,
            #                                "/".join(wdir.split("/")[-2:]),
            #                                thr[r_num],
            #                                scale,
            #                                trainer.model.results_dict['lambdas'][r_num],
            #                                r_ind,
            #                                results[0]['test_f1'],
            #                                results[0]['test_acc'],
            #                                results[0]['test_loss'],
            #                                trainer.model.results_dict['class_f1'][r_num],
            #                                trainer.model.results_dict['class_prec'][r_num],
            #                                trainer.model.results_dict['class_rec'][r_num],
            #                                trainer.model.results_dict['test_ver_ratio'][r_num],
            #                                trainer.model.results_dict['class_sup'][r_num]
            #                             ])
        else:
                                       
            ns_model = NsFrickerModel(output_dim, img_depth, lambdas, scale, thr, init_num_filt=init_num_filt, max_num_filt=max_num_filt, num_layers=n_layers, lr=lr, class_wts=class_weight_tensor)
            checkpoint_callback = ModelCheckpoint(monitor='val_f1',
                                                  auto_insert_metric_name=False,
                                                  save_top_k=1,
                                                  mode='max',
                                                  every_n_epochs=1,
                                                  filename="exp-epoch{epoch:02d}-val-f1_{val_f1:.2f}")
            trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], accelerator=device, devices=-1)
            trainer.fit(ns_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            results = trainer.test(ns_model, dataloaders=test_dataloader, ckpt_path="best")  # produces a list of length 1 containing a dictionary

            if args.log_file is not None:
               csv_wr_obj = mlg.Writer(args.log_file, ["tag", "exp_id", "rule", "seed", "batch_size", "lr", "epochs", "aux_data", "wdir", "thr", "scale", "lambda",
                                                       "rule_ind", "test_f1", "test_acc", "test_loss", "class_f1", "class_prec", "class_rec", "ver_ratio", "class_support"])

               # log can only store scalar values; have to store in model for arrays
               for r_num, r_ind in enumerate(trainer.model.results_dict['rule_ind']):               
                   csv_wr_obj.write_data([
                                            args.tag, 
                                            comet.experiment.id, 
                                            trainer.model.results_dict['r_names'][r_num], 
                                            seed, 
                                            batch_size, 
                                            lr, 
                                            epochs, 
                                            aux_data, 
                                            "/".join(wdir.split("/")[-2:]), 
                                            thr[r_num],
                                            scale, 
                                            trainer.model.results_dict['lambdas'][r_num],
                                            r_ind, 
                                            results[0]['test_f1'],
                                            results[0]['test_acc'], 
                                            results[0]['test_loss'],
                                            trainer.model.results_dict['class_f1'][r_num], 
                                            trainer.model.results_dict['class_prec'][r_num], 
                                            trainer.model.results_dict['class_rec'][r_num],
                                            trainer.model.results_dict['test_ver_ratio'][r_num], 
                                            trainer.model.results_dict['class_sup'][r_num]
                                         ])
       
        # make graphs
        #fig, axs = plt.subplots(2, 2)
        #fig.set_size_inches(w=8, h=8)
        #axs[0, 0].plot(range(len(ns_model.tot_loss)), ns_model.tot_loss, '-k')
        #axs[0, 0].set_title('Total Loss')
        #axs[0, 1].plot(range(len(ns_model.r_loss)), ns_model.r_loss, '-r')
        #axs[0, 1].set_title('Rule Loss')
        #axs[1, 0].plot([0.1 * x for x in range(11)], ns_model.f1_list, '-g')
        #axs[1, 0].set_title('Macro F1')
        #axs[1, 1].plot([0.1 * x for x in range(11)], ns_model.ver_rat_list, '-b')
        #axs[1, 1].set_title('Verification Ratio')
        #plt.savefig(model_type + "_" + "plot.png")

        # comet log
        fold_num = int(data_path[-1])
        if args.baseline:
            tag = ["base"]
        else:
            tag = ["nsf"]
        if args.rgb:
            tag += ["rgb"]
        if args.aux_data is not None:
            tag += [str(args.aux_data)]
        comet.experiment.add_tags(tag)
        if not args.baseline:
            comet.experiment.log_parameter("scale", scale)
            comet.experiment.log_parameter("aux_data", args.aux_data)
            comet.experiment.log_parameter("thr", str(args.thr))
            comet.experiment.log_parameter("lambdas", str(lambdas))
            fname = "preds_nsf_fold-" + str(fold_num) + "_expid-"+ comet.experiment.id + ".csv"
        else:
            fname = "preds_base_fold-" + str(fold_num) + "_expid-"+ comet.experiment.id + ".csv"

        comet.experiment.log_parameter("lr", lr)
        comet.experiment.log_parameter("epochs", epochs)        
        comet.experiment.log_parameter("batch_size", batch_size)        
        comet.experiment.log_parameter("wdir", "/".join(wdir.split("/")[-2:]))
        comet.experiment.log_code(file_name="train_pl.py")
        comet.experiment.log_code(file_name="ns_frickernet_pl.py")
        comet.experiment.log_code(file_name="frickernet_pl.py")
        comet.experiment.log_code(file_name="utils.py")
        comet.experiment.log_code(file_name="utils_learning.py")
        comet.experiment.log_code(file_name="test.py")
        comet.experiment.log_code(file_name="paths.py")
        comet.experiment.log_table(join(data_path, fname))
        comet.experiment.log_others(results[0])
        comet.experiment.log_other("fold", fold_num)
if __name__ == "__main__":
    main()

