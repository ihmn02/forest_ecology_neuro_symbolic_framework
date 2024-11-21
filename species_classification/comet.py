# import comet_ml at the top of your file
import os
from comet_ml import Experiment
from pars import args

#Create an experiment with your api key
experiment = Experiment(
    api_key=os.environ.get('COMET_KEY'),
    project_name="ns_spec_class",
    workspace="your_workspace", 
    disabled=not args.clog,
)
