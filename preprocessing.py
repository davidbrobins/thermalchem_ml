# Define functions to prepare chemical abundance calculations for input to ML models

# Import statements
import torch

# Rescale log(abundances) and log(T) to [-1, 1] interval
def scale_features(input_tensor):
    '''
    Inputs:
    input_tensor (torch.tensor): contains times, chemical abundances, and temperatures from solved chemical network
    Outputs:
    feature_tensor (torch.tensor): contains times, rescaled chemical abundances and temperatures
    '''

    # Create tensor to hold features
    feature_tensor = torch.zeros_like(input_tensor)
    # First column will still be sample index (these are not scaled)
    feature_tensor[:, 0] = input_tensor[:, 0]
    # Second column will still be times (these are not scaled)
    feature_tensor[:, 1] = input_tensor[:, 1]

    # Get number of columns in input_tensor
    num_cols = input_tensor.size()[-1]

    # Loop through sampled parameter (2-5) and abundance (6+) columns
    for i in range(2, num_cols):
        # Get the elements of the column, take logarithm
        col = torch.log10(input_tensor[:, i])

        # Get min and max values of log(abundance) or log(T)
        max = torch.max(col)
        min = torch.min(col)

        # Linearly rescale col to the interval [-1, 1]
        feature_tensor[:, i] = -1 + 2 * (col - min) / (max - min)

    # Return the tensor containing the features
    return feature_tensor