# Tools for generating 1D sample comparisons

import numpy as np # for math
import time # for timing
import pandas as pd # for dataframe handling
import torch # for neural networks
import matplotlib.pyplot as plt # for plotting
import os # make directories
import subprocess # to run bash commands
import preprocessing # functions to package data for input to emulator
import sampling_tools # functions to generate chempl input files
import architecture # defines the ML models used here
import model_training # functions to evaluate the emulator

# Function to load in optimal models
def load_optimal_models(model_path):
    '''
    Inputs: 
    model_path (str): path to directory where optimize_hyperparameter.py was run
    Outputs:
    encoder (Encoder): encoder trained with optimal hyperparameters
    decoder (Decoder): decoder trained with optimal hyperparameters
    time_emulator (TimeEmulator): latent-space time emulator trained with optimal hyperparameters
    '''
    # First, we need to read in the hyperparameter search results to get the best hyperparameter values
    hp_opt_results = pd.read_csv(model_path + 'hp_opt_results.csv')
    # Get row with minimal loss
    best_hps = hp_opt_results.loc[hp_opt_results['loss'].idxmin()]

    # Now create the model structures with these optimal hyperparams
    num_features = 573 # Number of features
    encoder = architecture.Encoder(num_features = num_features, latent_dim = best_hps['config/latent_dim'])
    decoder = architecture.Decoder(num_features = num_features, latent_dim = best_hps['config/latent_dim'])
    time_emulator = architecture.TimeEmulator(latent_dim = best_hps['config/latent_dim'], 
                                              hidden_layer_width = best_hps['config/hidden_layer_width'],
                                              num_hidden_layers = best_hps['config/num_hidden_layers'])

    # Read in the trained models
    encoder.load_state_dict(torch.load(model_path + 'encoder', weights_only = True))
    decoder.load_state_dict(torch.load(model_path + 'decoder', weights_only = True))
    time_emulator.load_state_dict(torch.load(model_path + 'time_emulator', weights_only = True))

    # Put all models in evaluation mode
    encoder.eval()
    decoder.eval()
    time_emulator.eval()

    return encoder, decoder, time_emulator


# Function 
def constant_params_test(T_gas, n_gas, G0_UV, cell_thickness, end_time, save_path, 
                         run_name, encoder, decoder, time_emulator, to_plot = True):
    '''
    Inputs:
    T_gas (float): gas temperature, in K
    n_gas (float): gas number density, in cm^-3
    G0_UV (float): radiation field strenght, unitless
    cell_thickness (float): cell size, in pc
    end_time (float): end time, in years
    save_path (str): directory path where results should be saved
    run_name (str): descriptive name for this run
    encoder (Encoder): trained instance of Encoder class in architecture.py
    decoder (Decoder): trained instance of Decoder class in architecture.py
    time_emulator (TimeEmulator): trained instance of TimeEmulator class in architecture.py
    to_plot (bool): whether or not to save an abundance evolution comparison plot (defaults to True)
    Outputs:
    chempl_timing (float): Chempl solving time (in s)
    ts_steps (list of ints): list of number of timesteps
    ts_losses (list of floats): emulator reconstruction loss corresponding to each entry in num_timesteps
    ts_timing (list of floats): emulator evaluation time corresponding to each entry in num_timesteps
    '''

    # Make a folder to save the results named run_name in save_path/
    os.makedirs(save_path + '/' + run_name, # Add a '/' for safety
                exist_ok = True) # Do not throw an error if the directory already exists

    # Chemical network calculation
    # Create chempl files with desired parameters
    sampling_tools.generate_chempl_files(T_gas, n_gas, G0_UV, cell_thickness,
                                         metallicity = 1, dir_path = save_path + run_name + '/', 
                                         t_max_year = end_time # Only run for 1 Myr
                                         )
    # Get correct run file
    run_file = save_path + run_name + '/run_file.dat'
    print('run file path: ', run_file)
    chempl_start = time.time() # Start timer
    # Run chempl on run_file, changing run location to chempl location (modify as needed for your working tree setup)
    chempl_run = subprocess.run(['./re', run_file], capture_output = True, text = True, 
                                cwd = '/home/drobinson/code/chempl/')
    # Print any errors
    print('Errors: ', chempl_run.stderr) 
    chempl_timing = time.time() - chempl_start # Save chempl runtime
    # Read in the abundances
    chempl_abuns = pd.read_csv(save_path + run_name + '/results.dat', sep = r'\s+')

    # Emulate the abundance trajectories

    # Modify chempl_abuns to match format expected for scaling
    # Ensure we have the right columns
    chempl_abuns.insert(0, 'sample_num', 0) # Put in 0 as the sample index 
    chempl_abuns.insert(1, 'G0_UV', G0_UV) # Insert G0_UV value
    chempl_abuns.insert(2, 'cell_thickness_pc', cell_thickness) # Insert cell thickness value (in pc)
    # Read in file of nontrivial abundances
    with open('nontrivial_abuns.txt', 'r') as file:
        non_trivial_abuns = file.read().splitlines()
    # Restrict abundances to only the nontrivial ones
    cols_to_keep = ['sample_num', 'Time', 'T_gas', 'n_gas', 'G0_UV', 'cell_thickness_pc'] + non_trivial_abuns
    chempl_abuns = chempl_abuns[cols_to_keep]
    # Set all abundances below floor to floor
    min_abun = 1e-20
    for col in chempl_abuns.columns[6:]: # Loop through abundance columns
        chempl_abuns[col] = chempl_abuns[col].where(chempl_abuns[col] > min_abun, min_abun)

    # Create structures to store values from the emulator
    # Create a dataframe to store emulated abundances
    emulated_abuns = chempl_abuns.iloc[[0]] # Abundances at t=0 are just the true initial ones
    # Create a blank shell for the losses at each timestep
    losses = np.ndarray(shape = len(chempl_abuns))
    # Get an array of the time values
    times = chempl_abuns['Time'].values

    # Set pytorch device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device") # Print device being used

    # Loop through times (after t = 0) at the end of each timestep
    for i in range(1, len(times)):
        delta_t = times[i] - times[i - 1] # Calculate timestep

        # Evaluate the emulator
        # Get abundances at the beginning of the timestep
        initial = torch.from_numpy(emulated_abuns.iloc[i - 1].values).to(dtype = torch.float32).unsqueeze(0) # Note: we are applying the emulator iteratively
        # Scale the features
        scaled_initial = preprocessing.scale_features(initial)
        # Extract the abundances and features
        initial_abuns = scaled_initial[:, 6:].to(device) # First 2 columns are index, time, next 4 are physical features, rest are scaled abundances
        physical_params = scaled_initial[:, 2:6].to(device)
        # Encode the abundances
        initial_latents = encoder(initial_abuns)
        # Combine the latents and physical params to pass to the time emulator
        initial_for_te = torch.cat((physical_params, initial_latents), dim = 1)
        # Put timestep into a tensor
        dt_tensor = torch.tensor(delta_t, device = 'cpu', dtype = torch.float32).unsqueeze(0).unsqueeze(0)
        # Apply the latent-space time emulator
        evolved_latents = model_training.iterate_time_emulator_twice(time_emulator, initial_for_te, 
                                                                     dt = dt_tensor, device = device)
        # Decode the evolved latents
        pred_scaled_abuns = decoder(evolved_latents)
        
        # Calculate the loss
        # Get the true values after the timestep from chempl_abuns
        target = torch.from_numpy(chempl_abuns.iloc[i].values).to(dtype = torch.float32).unsqueeze(0)
        # Scale these target
        scaled_target = preprocessing.scale_features(target)
        # Extract the (scaled) abundances
        true_scaled_abuns = scaled_target[:, 6:].to(device) 
        loss_func = model_training.ReconstructionLoss() # Select reconstruction loss
        loss = loss_func(pred_scaled_abuns, true_scaled_abuns) # Calculate loss
    
        # Decode abundances
        pred_abuns = 10 ** (-10 + 10 * pred_scaled_abuns) # Assumes min_abun = 1e-20
        # Convert to a dataframe
        pred_abuns = pd.DataFrame(chempl_abuns.iloc[i].to_list()[:6] + [x.item() for x in pred_abuns[0]]).transpose()
        # Name columns to match abundance dataframes
        pred_abuns.columns = emulated_abuns.columns
    
        # Put predicted abundances, loss into appropriate structures
        emulated_abuns = pd.concat([emulated_abuns, pred_abuns], ignore_index = True)
        losses[i] = loss.item()

    if to_plot == True: # if saving plot, create the plot
        plt.plot(chempl_abuns['Time'], chempl_abuns['H2'], linestyle = 'solid', color = 'darkorange', label = 'H2')
        plt.plot(emulated_abuns['Time'], emulated_abuns['H2'], linestyle = 'dashed', color = 'darkorange')
        plt.plot(chempl_abuns['Time'], chempl_abuns['CO'], linestyle = 'solid', color = 'steelblue', label = 'CO')
        plt.plot(emulated_abuns['Time'], emulated_abuns['CO'], linestyle = 'dashed', color = 'steelblue')
        plt.plot(chempl_abuns['Time'], chempl_abuns['HCN'], linestyle = 'solid', color = 'green', label = 'HCN')
        plt.plot(emulated_abuns['Time'], emulated_abuns['HCN'], linestyle = 'dashed', color = 'green')
        plt.plot(chempl_abuns['Time'], chempl_abuns['HCO+'], linestyle = 'solid', color = 'purple', label = 'HCO+')
        plt.plot(emulated_abuns['Time'], emulated_abuns['HCO+'], linestyle = 'dashed', color = 'purple')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time [yr]')
        plt.ylabel('Abundance')
        plt.legend()
        plt.savefig(save_path + run_name + '/abundance_evolution.pdf', bbox_inches = 'tight')
        plt.close()

    # Now compare loss over whole evolution with different numbers of emulator timesteps
    ts_steps = [1, 2, 4, 8, 16, 32, 64, 128] # Number of timesteps to test
    ts_losses = [] # Array for losses vs. number of timesteps
    ts_timings = [] # Array for timing vs. number of timesteps
    for num_timesteps in ts_steps: # Loop through timesteps
        print('Number of timesteps: ', num_timesteps)
        start_time = time.time() # Timing
        total_time = times[-1] # Total time (get from chempl results above)
        timestep = total_time / num_timesteps # Determine length of each timestep
        emulator_time = 0 # Counter for the time
        
        # Get abundances at the beginning of the timestep
        initial = torch.from_numpy(chempl_abuns.iloc[0].values).to(dtype = torch.float32).unsqueeze(0) 
        # Scale the features
        scaled_initial = preprocessing.scale_features(initial)
        # Extract the abundances and features
        initial_abuns = scaled_initial[:, 6:].to(device) # First 2 columns are index, time, next 4 are physical features, rest are scaled abundances
        physical_params = scaled_initial[:, 2:6].to(device)
        # Encode the abundances
        initial_latents = encoder(initial_abuns)
        dt_tensor = torch.tensor(timestep, device = 'cpu', dtype = torch.float32).unsqueeze(0).unsqueeze(0)  
        while emulator_time < total_time: # Loop through timesteps
            print('Start of timestep: ', emulator_time)
            # Combine the latents and physical params to pass to the time emulator
            initial_for_te = torch.cat((physical_params, initial_latents), dim = 1)
            # Iterate the latents forward
            evolved_latents = model_training.iterate_time_emulator_twice(time_emulator, initial_for_te, 
                                                                         dt = dt_tensor, device = device)
            initial_latents = evolved_latents # Initial latents for next timestep are evolved ones from this timestep
            emulator_time += timestep # Iterate time forward
            print('End of timestep: ', emulator_time)
        # Decode the final evolved latents
        pred_scaled_abuns = decoder(evolved_latents)
        # Calculate the final loss
        # Get target from final line of chempl_abuns
        target = torch.from_numpy(chempl_abuns.iloc[-1].values).to(dtype = torch.float32).unsqueeze(0)
        # Scale the features
        scaled_target = preprocessing.scale_features(target)
        # Extract the abundances and features
        true_scaled_abuns = scaled_target[:, 6:].to(device) # First 2 columns are index, time, next 4 are physical features, rest are scaled abundances
        loss_func = model_training.ReconstructionLoss()
        loss = loss_func(pred_scaled_abuns, true_scaled_abuns)
        step_time = time.time() - start_time # timing

        # Put results in appropriate arrays
        ts_losses.append(loss.item())
        ts_timings.append(step_time)

    return chempl_timing, ts_steps, ts_losses, ts_timings
    
    