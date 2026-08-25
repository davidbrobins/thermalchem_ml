# Script to optimize model hyperparameters
# Usage: python optimize_hyperparams.py samples_dir/ num_runs

# Import statements
from ray import tune # Use Ray Tune for hyperparameter search
from ray.tune.search.optuna import OptunaSearch # Use Optuna for the hyperparameter optimization algorithm
import sys # Command line arguments
import pandas as pd # Pandas for dataframe handling
import numpy as np
import matplotlib.pyplot as plt # Plotting
# Use ML pipeline set up here
import architecture # ML model architecture
import model_training # Training
import preprocessing # Feature scaling

# Define the search space
search_space = {'learning_rate' : tune.loguniform(1e-3, 1e-1), # Learning rate for training
                'latent_dim' : tune.randint(6, 31), # Dimension of autoencoder latent space
                'hidden_layer_width' : tune.randint(10, 41), # Width of time emulator hidden layers
                'num_hidden_layers' : tune.randint(2, 7) # Number of hidden layers in time emulator
                }

# Load in data, perform train-test-validate split to get validation data

# Define objective function (requires the training data)
def objective(config):
    # Training on validation set here (5-fold split? might be computationally difficult)
    return {'loss': config['latent_dim'] + (config['hidden_layer_width'] * config['learning_rate']) + 
            config['num_hidden_layers']}

# Set up the hyperparameter search
tuner = tune.Tuner(objective, # Function reporting error
                   tune_config = tune.TuneConfig(metric = 'loss', mode = 'min', num_samples = 100, 
                                                 search_alg = OptunaSearch()), 
                   param_space = search_space) # Search space

# Perform the hyperparameter search
results = tuner.fit()

# Get the best hyperparameters
best_hyperparams = results.get_best_result(metric = 'loss', mode = 'min').config
print(best_hyperparams)

# Inspect the results
results_df = results.get_dataframe() # Package as a dataframe
# Save relevant columns
# Plot loss vs. each search parameter
