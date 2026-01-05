# Define functions to prepare chemical abundance calculations for input to ML models

# Import statements
import torch
import numpy as np

# Rescale log(T_gas), log(n_gas), log(G0_UV), log(cell_thickness_pc), and log(abundances) to [-1, 1] interval
def scale_features(input_tensor, T_gas_min = 1e1, T_gas_max = 1e4, n_gas_min = 1e1, 
                   n_gas_max = 1e6, G0_UV_min = 1e-2, G0_UV_max = 1e3, min_abundance = 1e-20):
    ''' 
    Inputs:
    input_tensor (torch.tensor): contains times, chemical abundances, and temperatures from solved chemical network
    T_gas_min (float): Minimum temperature (in K) in the sampling scheme, defaults to 10
    T_gas_max (float): Maximum temperature (in K) in the sampling scheme, defaults to 10^4
    n_gas_min (float): Minimum density (in cm^-3) in the sampling scheme, defaults to 10
    n_gas_max (float): Maximum density (in cm^-3) in the sampling scheme, defaults to 10^6
    G0_UV_min (float): Minimum UV radiation field strength in the sampling scheme, defaults to 10^-2
    G0_UV_max (float): Maximum UV radiation field strength in the sampling scheme, defaults to 10^3
    min_abundance (float): Minimum abundance [log(n_i/n_gas)] used to filter chempl data
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


    # Scale the features
    # Scale temperature column (index 2)
    log_T = torch.log10(input_tensor[:, 2]) # Get log(T/K)
    # Get min, max temps
    log_T_min = np.log10(T_gas_min)
    log_T_max = np.log10(T_gas_max)
    # Linearly rescale to the interval [-1, 1]
    feature_tensor[:, 2] = -1 + 2 * (log_T - log_T_min) / (log_T_max - log_T_min)
    # Do the same for density (index 3)
    log_n = torch.log10(input_tensor[:, 3])
    log_n_min = np.log10(n_gas_min)
    log_n_max = np.log10(n_gas_max)
    feature_tensor[:, 3] = -1 + 2 * (log_n - log_n_min) / (log_n_max - log_n_min)
    # Do the same for G0_UV (index 4)
    log_G = torch.log10(input_tensor[:, 4])
    log_G_min = np.log10(G0_UV_min)
    log_G_max = np.log10(G0_UV_max)
    feature_tensor[:, 4] = -1 + 2 * (log_G - log_G_min) / (log_G_max - log_G_min)
    # Do the same for cell_thickness_pc (index 5)
    log_ct = torch.log10(input_tensor[:, 5])
    # Values are sampled between [1e-2, 1e1] * Jeans length, so need min/max Jeans length values
    jl_min = 17 * np.sqrt(T_gas_min / n_gas_max) # Reached with min T value, max n value
    jl_max = 17 * np.sqrt(T_gas_max / n_gas_min) # Reached with max T value, min n value
    # Scale these to get min, max possible cell thickness values
    log_ct_min = np.log10(jl_min / 100)
    log_ct_max = np.log10(jl_max * 10)
    feature_tensor[:, 5] = -1 + 2 * (log_ct - log_ct_min) / (log_ct_max - log_ct_min)
    # Scale abundances (indices 6+)
    # Min and max values are set by abundance filter, 1
    log_y_min = np.log10(min_abundance)
    log_y_max = np.log10(1)
    # Loop through abundance columns (indices 6+)
    for i in range(6, num_cols):
        # Get log(n_i/n_gas)
        log_y = torch.log10(input_tensor[:, i]) 
        # Linearly rescale col to the interval [-1, 1]
        feature_tensor[:, i] = -1 + 2 * (log_y - log_y_min) / (log_y_max - log_y_min)

    # Return the tensor containing the features
    return feature_tensor