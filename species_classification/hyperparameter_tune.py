import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ax.plot.contour import plot_contour
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render

from utils import Hsi_raster, trans
from ns_frickernet_pl import NsFrickerModel
from pars import args as cl_args

import pandas as pd
import numpy as np
import uuid
from tabulate import tabulate
from sklearn.model_selection import ParameterGrid
import os
import io
from contextlib import redirect_stdout
import json
import base64
import plotly.io as pio

pio.renderers.default = "png"


def base64_to_bytes(io_str, fname):
    # render outputs a json string with base64 encoded image
    # this function turns that base64 into a png file
    json_string = io_str.getvalue()
    json_string = json_string.replace("'", "\"")
    json_obj = json.loads(json_string)
    png_bytes = base64.b64decode(json_obj['image/png'])

    f = open(fname, 'wb')
    f.write(png_bytes)
    f.close()

param_grid = {'r1':np.linspace(0.01, 20., 8), 'r2':np.linspace(0.01, 20., 8)}
gs_pts  = list(ParameterGrid(param_grid))
real_r1s = []
real_r2s = []

def alter_search(func):
    return_func = func
    def grid_search_wrapper(*args, **kwargs):
        print('\n Grid search: ')
        print(f'  original values: {args[0]}')
        params = gs_pts.pop(0)
        print(f'  new values: {params}')
        real_r1s.append(params['r1'])
        real_r2s.append(params['r2'])
        return func(params)

    def random_search_wrapper(*args, **kwargs):
        print('\nRandom search: ')
        print(f'  original values: {args[0]}')
        r1_lambda = (20.0 - 0.01) * np.random.random() + 0.01
        r2_lambda = (20.0 - 0.01) * np.random.random() + 0.01
        params = {'r1': r1_lambda, 'r2': r2_lambda}
        print(f'  new values: {params}')
        real_r1s.append(r1_lambda)
        real_r2s.append(r2_lambda)
        return func(params)

    def bayesian_search_wrapper(*args, **kwargs):
        print('\nBayesian Search:')
        print(f'  original values: {args[0]}')
        real_r1s.append(args[0]['r1'])
        real_r2s.append(args[0]['r2'])
        print(f'  new_values: {args[0]}')
        return func(args[0])

    if (cl_args.htune == 'gs'):
        return_func = grid_search_wrapper
    elif (cl_args.htune == 'r'):
        return_func = random_search_wrapper
    elif (cl_args.htune == 'b'):
        return_func = bayesian_search_wrapper
 
    return return_func

def run_ax(device, train_dataloader, val_dataloader, test_dataloader, n_layers, output_dim, init_num_filt, max_num_filt, img_depth, lambdas, thr, scale, lr, epochs, n_trials, seed, class_weight_tensor):
    test_f1s = []
    r1_class_f1s = []
    r2_class_f1s = []
    results_df_fname = 'htune_results_v3.csv'

    @alter_search
    def train_evaluate(parameterization):
        rule_lambdas = [
                            parameterization['r1'],
                            parameterization['r2'],
                            #parameterization['r3']
        ]
        print(f' \n\nrule lambdas = {rule_lambdas}\n\n')
        ns_model = NsFrickerModel(output_dim, img_depth, rule_lambdas, scale, thr, init_num_filt=init_num_filt, max_num_filt=max_num_filt, num_layers=n_layers, lr=lr, class_wts=class_weight_tensor)
        checkpoint_callback = ModelCheckpoint(monitor='val_f1',
                                              auto_insert_metric_name=False,
                                              save_top_k=1,
                                              mode='max',
                                              every_n_epochs=1,
                                              filename="exp-epoch{epoch:02d}-val-f1_{val_f1:.2f}")
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], accelerator=device, devices=-1)
        trainer.fit(ns_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        val_f1 = trainer.validate(ns_model, dataloaders=val_dataloader, ckpt_path="best")[0]['val_f1']  # produces a list of length 1 containing a dictionary 


        # to collect data for publicaiton
        test_f1 = trainer.test(ns_model, dataloaders=test_dataloader, ckpt_path="best")[0]['test_f1'] 
        r1_class_f1 = trainer.model.results_dict['class_f1'][0]
        r2_class_f1 = trainer.model.results_dict['class_f1'][1]
        test_f1s.append(test_f1)
        r1_class_f1s.append(r1_class_f1)
        r2_class_f1s.append(r2_class_f1)
        
        # calc l2 norm of rule weights
        r = torch.tensor(rule_lambdas)
        l2norm = torch.sqrt(torch.sum(torch.square(r)))
        return {"val_f1": (val_f1, 0.0), "l2norm": (l2norm, 0.0)}


    ## Run the optimization
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "r1", "type": "range", "bounds": [0.01, 20.0], "log_scale": False},
            {"name": "r2", "type": "range", "bounds": [0.01, 20.0], "log_scale": False},
            #{"name": "r3", "type": "range", "bounds": [0.01, 33.3], "log_scale": True}
            ### extend this dictionary to other hyperparameters you want to use which are forwarded to the evaluation_function
            #{"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            #{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
            #{"name": "stepsize", "type": "range", "bounds": [20, 40]},
            #{"name": "batchsize", "type": "range", "bounds": [10, 100]},
            
        ],
        #parameter_constraints=[
        #    "r1 + r2 <= 20.0"
        #],
        #outcome_constraints=["l2norm <= 20."],
        evaluation_function=train_evaluate,
        objective_name='val_f1',
        minimize=False,
        total_trials=n_trials,
    )
    
    if (cl_args.htune == 'b'):
        # only save these images for bayesian optimization
        # get plots from trials
        f1 = io.StringIO()
        with redirect_stdout(f1):
            # capture base64 output
            render(plot_contour(model=model, param_x="r1", param_y="r2", metric_name="val_f1"))

        f2 = io.StringIO()
        with redirect_stdout(f2):
            # capture base64 output
            render(plot_slice(model, "r1", "val_f1")) 

        f3 = io.StringIO()
        with redirect_stdout(f3):
            render(plot_slice(model, "r2", "val_f1"))

        
        best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
        best_objective_plot = optimization_trace_single_method(
            y=np.maximum.accumulate(best_objectives, axis=1),
            title="Model performance vs. # of iterations",
            ylabel="val. F1",
        )

        f4 = io.StringIO()
        with redirect_stdout(f4):
            # capture base64 output
            render(best_objective_plot)
    
        base64_to_bytes(f1, "val_f1_contour_plot.png")
        base64_to_bytes(f2, "r1_trace.png")
        base64_to_bytes(f3, "r2_trace.png")
        base64_to_bytes(f4, "objective_vs_iters_plot.png")
    

    # dataframe with trial results
    results_df = exp_to_df(experiment)

    # add f1s and seed as columns
    results_df.loc[:, 'test_f1'] = test_f1s
    results_df.loc[:, 'r1_class_f1'] = r1_class_f1s
    results_df.loc[:, 'r2_class_f1'] = r2_class_f1s
    results_df.loc[:, 'seed'] = seed
    results_df.loc[:, 'search_type'] = cl_args.htune
    results_df.loc[:, 'exp_name'] = str(uuid.uuid4())

    # update r1 and r2 values when bayesian opt. not used
    results_df.loc[:, 'r1_actual'] = real_r1s
    results_df.loc[:, 'r2_actual'] = real_r2s

    # order columns
    col_order = [
                   "trial_index", 
                   "arm_name", 
                   "trial_status", 
                   "generation_method", 
                   "seed", 
                   "exp_name",
                   "search_type",
                   "l2norm",  
                   "r1",
                   "r1_actual", 
                   "r2", 
                   "r2_actual",
                   "test_f1",
                   "val_f1",
                   "r1_class_f1",
                   "r2_class_f1"
    ]

    print_cols = [
                   "trial_index", 
                   "trial_status",  
                   "seed", 
                   "search_type",
                   "r1",
                   "r1_actual", 
                   "r2", 
                   "r2_actual",
                   "test_f1",
                   "val_f1",
                   "r1_class_f1",
                   "r2_class_f1"
    ] 
    results_df = results_df.loc[:, col_order]

    results_df_sort = results_df.sort_values(by="val_f1", ascending=False)
    print(tabulate(results_df_sort.loc[:, print_cols], headers='keys', tablefmt='psql'))

    # save results dataframe to file
    if (os.path.exists(results_df_fname)):
        results_df.to_csv(results_df_fname, header=False, index=False, mode='a')
    else:
        results_df.to_csv(results_df_fname, header=True, index=False) 
