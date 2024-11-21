from deepforest import get_data
from deepforest import evaluate
from deepforest import main
from deepforest import ns_deepforest as nsd
import random
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from pars import args
from hyperparameter_tune import run_ax

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

scale = args.scale
rule_lambdas = args.lambdas
baseline_model = args.baseline
n_trials = args.n_trials
epochs = args.epochs


print('\n\nCommand line args:')
print('  epochs: {}'.format(epochs))
print('  hyperparameter tune:? {}'.format(args.htune))
print('  lambdas: {}'.format(rule_lambdas))
print('  n_trials: {}'.format(n_trials))    
print('  seed: {}'.format(seed))
print('  scale: {}'.format(scale))
print('  baseline model: {}'.format(baseline_model))
print('\n\n')
 

training_file = get_data("datasets/niwo/training4/NIWO-train.csv")
validation_file = get_data("datasets/niwo/training4/NIWO-val.csv")
test_file = get_data("datasets/niwo/evaluation4/NIWO-test.csv")

if (not args.htune):
    if (not baseline_model):
        model = nsd.Ns_deepforest(scale=scale, rule_lambdas=rule_lambdas)  #main.deepforest()
    else:
        model = main.deepforest()

    model.use_release()   # uses pre-trained weights

    model.config["batch_size"] = 1
    model.config["train"]["csv_file"] = training_file
    model.config["validation"]["csv_file"] = validation_file

    model.config["train"]["root_dir"] = os.path.dirname(training_file)
    model.config["validation"]["root_dir"] = os.path.dirname(validation_file)

    model.config["train"]["epochs"] = epochs
    model.config["save-snapshot"] = False

    #callback = ModelCheckpoint(dirpath='tmp',
    #                                 monitor='box_recall', 
    #                                 mode="max",
    #                                 save_top_k=1,
    #                                 filename="box_recall-{epoch:02d}-{box_recall:.2f}")

    model.create_trainer()
    #model.config["train"]["fast_dev_run"] = True
    model.trainer.fit(model)
    #results = model.trainer.validate(model)
    root_dir = os.path.dirname(test_file)
    results = model.evaluate(test_file, root_dir, iou_threshold = 0.4)
    print(f'\nbbox_precision: {results["box_precision"]}')
    print(f'bbox_recall: {results["box_recall"]}')
    print(f'f1: {results["f1"]}\n')

else:
    # run_ax(device, lambdas, training_file, val_file, test_file, batch_size, scale, lr, epochs, n_trials, seed)
    run_ax('auto', rule_lambdas, training_file, validation_file, test_file, 1, scale, 0.001, epochs, n_trials, seed)

#sample_image_path = get_data("OSBS_029.png")
#img = model.predict_image(path=sample_image_path, return_plot=True)
