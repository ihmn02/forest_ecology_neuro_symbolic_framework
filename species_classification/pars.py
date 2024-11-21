import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rgb', action='store_true', help='use rgb data')
parser.add_argument('--wdir', required=True, help='working directory')
parser.add_argument('--baseline', action='store_true', help='use baseline model')
parser.add_argument('--htune', default=None, choices=['b', 'gs', 'r'],  required=False, help='tune hyperparameters using random search, grid search, or bayesian optimization')
parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train")
parser.add_argument('--batch' , type=int, default=32, help="batch size")
parser.add_argument('--clog', action='store_true', help='log to comet')
parser.add_argument('--aux_data', type=str, default=None, help='auxilliary data to use added to rgb as extra channel')
parser.add_argument('--thr', type=float, nargs='+', help='aux. data threshold')
parser.add_argument('--lambdas', type=float, nargs='+', help='rule lambdas')
parser.add_argument('--scale', type=float, default=-1, help='coefficient for total rule loss')
parser.add_argument('--log_file', type=str, default=None, help='local log filename')
parser.add_argument('--lr', type=float, default=1e-4, required=True, help='learning rate')
parser.add_argument('--tag', type=str, required=True, default=None, help='tag for csv log')
parser.add_argument('--seed', type=int, required=False, default=42, help='RNG seed')
parser.add_argument('--n_trials', type=int, required=False, default=5, help='number of hyperparameter tuning trials')
args = parser.parse_args()
