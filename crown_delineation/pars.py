import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--htune', default=None, choices=['b', 'gs', 'r'],  required=False, help='tune hyperparameters using random search, grid search, or bayesian optimization')
parser.add_argument('--seed', type=int, required=False, default=42, help='RNG seed')
parser.add_argument('--n_trials', type=int, required=False, default=5, help='number of hyperparameter tuning trials')
parser.add_argument('--lambdas', type=float, nargs='+', help='rule lambdas')
parser.add_argument('--scale', type=float, default=1.0, help='coefficient for total rule loss')
parser.add_argument('--baseline', action='store_true', help='use baseline model')
parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train")
args = parser.parse_args()
