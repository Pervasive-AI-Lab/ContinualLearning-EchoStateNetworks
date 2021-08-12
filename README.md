# Continual Learning with Echo State Networks
Experiments for ESANN 2021

To reproduce experiments: `python launch_experiment.py --config_file CONFIGS/path/to/file.yaml`.  
If you do not want to use `ray` to run your experiments (`pip install ray`) you can set the `no_ray` flag to `true` in the configuration file.

You can download the SSC dataset from [Kaggle](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset). In the experiment configuration file, the `dataroot` argument should point to the folder containing all the 30 SSC folders.
