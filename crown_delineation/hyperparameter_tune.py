import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ax.plot.contour import plot_contour
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render

from deepforest import get_data
from deepforest import evaluate
from deepforest import main
from deepforest import ns_deepforest as nsd

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import ParameterGrid
import uuid
import os
import io
from contextlib import redirect_stdout
import json
import base64
import plotly.io as pio
pio.renderers.default = "png"
#init_notebook_plotting() # for jupyter notebook plots 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pars import args as cl_args

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

def save_plot_obj_data(model, out_fname):
    """
    this function pulls data from ax plot_obj allowing the data to be used outside of ax
    """
    plot_obj = plot_contour(model=model, param_x="r1", param_y="r2", metric_name="val_f1")
    x = plot_obj._asdict()['data']['data'][0]['x']    # 0 is metric; 1 is std dev.
    y = plot_obj._asdict()['data']['data'][0]['y']   

    Z = np.array(plot_obj._asdict()['data']['data'][2]['z'])


    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Val F1 ')
    fig.savefig(out_fname)

param_grid = {'r1':np.linspace(0.01, 9.0, 8), 'r2':np.linspace(0.01, 9.0, 8)}
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
        r1_lambda = (9.0 - 0.01) * np.random.random() + 0.01
        r2_lambda = (9.0 - 0.01) * np.random.random() + 0.01
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


def run_ax(device, lambdas, train_file, val_file, test_file, batch_size, scale, lr, epochs, n_trials, seed):   
    val_precs = []
    val_recs = []
    test_precs = []
    test_recs = []
    test_f1s = []
    results_df_fname = 'htune_results_v1.csv' 

    @alter_search
    def train_evaluate(parameterization):
        rule_lambdas = [
                            parameterization['r1'],
                            parameterization['r2'],
        ]
        print(f' \n\nrule lambdas = {rule_lambdas}\n\n')
        model = nsd.Ns_deepforest(scale=scale, rule_lambdas=rule_lambdas)
        model.use_release()   # uses pre-trained weights
        model.config["batch_size"] = batch_size
        model.config["train"]["epochs"] = epochs
        model.config["save-snapshot"] = False

        # datasets
        model.config["train"]["csv_file"] = train_file
        model.config["validation"]["csv_file"] = val_file
        model.config["train"]["root_dir"] = os.path.dirname(train_file)
        model.config["validation"]["root_dir"] = os.path.dirname(val_file)

        model.create_trainer()
        model.trainer.fit(model)

        root_dir = os.path.dirname(val_file) 
        val_results = model.evaluate(val_file, root_dir, iou_threshold = 0.4)  # produces a list of length 1 containing a dictionary 
        val_f1 = val_results['f1']


        # to collect data for publicaiton
        val_precs.append(val_results['box_precision'])
        val_recs.append(val_results['box_recall'])

        root_dir = os.path.dirname(test_file)
        test_results = model.evaluate(test_file, root_dir, iou_threshold = 0.4)
        
        test_f1s.append(test_results['f1'])
        test_precs.append(test_results['box_precision'])
        test_recs.append(test_results['box_recall'])

        # calc l2 norm of rule weights
        r = torch.tensor(rule_lambdas)
        l2norm = torch.sqrt(torch.sum(torch.square(r)))
        return {"val_f1": (val_f1, 0.0), "l2norm": (l2norm, 0.0)}


    ## Run the optimization
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "r1", "type": "range", "bounds": [0.01, 9.0], "log_scale": False},
            {"name": "r2", "type": "range", "bounds": [0.01, 9.0], "log_scale": False},
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
        # get images
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
    

    # trial data
    results_df = exp_to_df(experiment)

    # add f1s and seed as columns
    results_df.loc[:, 'test_f1'] = test_f1s
    results_df.loc[:, 'test_prec'] = test_precs
    results_df.loc[:, 'test_rec'] = test_recs
    results_df.loc[:, 'val_prec'] = val_precs
    results_df.loc[:, 'val_rec'] = val_recs
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
                   "test_prec",
                   "test_rec",
                   "test_f1",
                   "val_prec",
                   "val_rec",
                   "val_f1"
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
                   "val_f1"
    ] 
    results_df = results_df.loc[:, col_order]

    results_df_sort = results_df.sort_values(by="val_f1", ascending=False)
    print(tabulate(results_df_sort.loc[:, print_cols], headers='keys', tablefmt='psql'))

    # save results dataframe to file
    if (os.path.exists(results_df_fname)):
        results_df.to_csv(results_df_fname, header=False, index=False, mode='a')
    else:
        results_df.to_csv(results_df_fname, header=True, index=False) 
    results_df = results_df.sort_values(by="val_f1", ascending=False)
    print(tabulate(results_df, headers='keys', tablefmt='psql'))
