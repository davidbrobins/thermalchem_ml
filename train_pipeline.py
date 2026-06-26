# Script to train the full emulator pipeline
# Usage: python train_pipeline.py samples_dir/ latent_dim hidden_layer_width num_hidden_layers substeps

# Import statements
import sys # Command line arguments
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import torch # Torch for ML
import itertools # Iteration tools
import matplotlib.pyplot as plt # Plotting
# Use ML pipeline set up here
import architecture # ML model architecture
import model_training # Training steps
import preprocessing # Feature scaling

# Unpack command line arguments
# Name of this file, path location of data, model properties
(pyfilename, samples_dir, num_runs, latent_dim, hidden_layer_width, num_hidden_layers) = sys.argv 
# Convert the model properties to integers
num_runs = int(num_runs)
latent_dim = int(latent_dim)
hidden_layer_width = int(hidden_layer_width)
num_hidden_layers = int(num_hidden_layers)

# Torch set up
# Check if GPU is available (use CPU if not)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device") # Print device being used
# Set number of threads used to 8
torch.set_num_threads(8)
# Print the number of CPUs seen by torch to confirm
print('CPUs to be used by torch:', torch.get_num_threads())

# Read in the chemical network data
# Create a list to hold all the dataframes
dfs_list = []
# Loop through chemical network samples
for index in range(num_runs):
    # Read in the results file as a pandas dataframe
    chempl_results = pd.read_csv(samples_dir + str(index).zfill(6) + '/results.dat',
                                 sep = r'\s+')
    # Read in the sample parameters
    sampled_params = pd.read_csv(samples_dir + str(index).zfill(6) + '/sampled_params.dat',
                                 sep = r'\s+', header = None, index_col = 0).squeeze("columns")
    # Now add the sampled parameters to the dataframe (n_gas, T_gas are already included)
    chempl_results.insert(4, 'G0_UV', sampled_params['G0_UV'])
    chempl_results.insert(5, 'cell_thickness_pc', sampled_params['cell_thickness_pc'])
    # Also add the sample number as the first column
    chempl_results.insert(0, 'sample_num', index)
    # Add the dataframe to the running list
    dfs_list.append(chempl_results)
# Concatenate all the datafames
all_data = pd.concat(dfs_list)
# Look at where the data is under some threshold
min_abundance = 1e-20
# Only keep abundance columns with some values above this threshold
nonzero_abuns = all_data[all_data.columns[7:]].loc[:, (all_data > min_abundance).any()]
# Check if each abundance is below min_abundance, and increase abundance to the threshold if so
filtered_data = nonzero_abuns.where(nonzero_abuns > min_abundance, min_abundance)
# Add back the sample number
filtered_data.insert(loc = 0, column = 'sample_num', value = all_data['sample_num'])
# Add back the time column
filtered_data.insert(loc = 1, column = 'Time', value = all_data['Time'])
# Add back the sampled parameters
filtered_data.insert(loc = 2, column = 'T_gas', value = all_data['T_gas'])
filtered_data.insert(loc = 3, column = 'n_gas', value = all_data['n_gas'])
filtered_data.insert(loc = 4, column = 'G0_UV', value = all_data['G0_UV'])
filtered_data.insert(loc = 5, column = 'cell_thickness_pc', value = all_data['cell_thickness_pc'])

# Separate pairs of time points with same parameters into "initial" and "final" datasets
# Create a list to hold all the dataframes
initial_dfs_list = []
final_dfs_list = []
# Define parameters for selecting time-pairs 
min_delta_t = 10 # Minimum delta_t value, in years (yr)
num_pairs_per_run = 1000 # Number of pairs per run
# Loop through chempl runs
for sample_index in range(num_runs): 
    # Get (filtered) data just for that run
    filtered_run = filtered_data.loc[filtered_data['sample_num'] == sample_index]
    # Get all pairs of indices (with earlier time first)
    index_pairs = np.array(list(itertools.combinations(filtered_run.index, 2)))
    # Get first and second elements of each pair
    first_index = index_pairs[:, 0]
    second_index = index_pairs[:, 1]
    # Get delta_t (in years) for each pair (neeed to reset indices to allow subtraction)
    delta_t_vals = filtered_run['Time'].iloc[second_index].reset_index(drop = True) - filtered_run['Time'].iloc[first_index].reset_index(drop = True)
    # Get indices for pairs where delta_t is above the minimum threshold defined above
    restricted_index = delta_t_vals.loc[delta_t_vals > min_delta_t].index
    # Choose num_pairs_per_run random indices where delta_t is above min_delta_t (without replacement to avoid duplicate pairs)
    random_index = np.random.choice(restricted_index, size = num_pairs_per_run, replace = False)
    # Now restrict pair indices to pairs with delta_t above minimum threshold
    first_index = first_index[random_index]
    second_index = second_index[random_index]
    # Generate dataframes with the rows at these indices
    initial_data = filtered_run.iloc[first_index]
    final_data = filtered_run.iloc[second_index]
    # Reset indices to default so can subtract
    initial_data = initial_data.reset_index(drop = True)
    final_data = final_data.reset_index(drop = True)
    # Get time difference
    delta_t = final_data['Time'] - initial_data['Time']
    # Put this as the time column in both dataframes
    initial_data['Time'] = delta_t
    final_data['Time'] = delta_t
    # Append each dataframe to the appropriate list
    initial_dfs_list.append(initial_data)
    final_dfs_list.append(final_data)
# Combine the dataframes in each list
all_initial_data = pd.concat(initial_dfs_list)
all_final_data = pd.concat(final_dfs_list)
# Convert these dataframes to torch tensors
initial_tensor = torch.from_numpy(all_initial_data.values).to(dtype = torch.float32)
final_tensor = torch.from_numpy(all_final_data.values).to(dtype = torch.float32)
# Scale them to the interval [-1, 1] using the function from preprocessing.py
scaled_initial = preprocessing.scale_features(initial_tensor)
scaled_final = preprocessing.scale_features(final_tensor)

# Package data in the AbunAfterDt class defined in preprocessing.py
# Get needed tensors
initial_features = scaled_initial[:, 2:] # Keep physical parameters @ indices 2-5, abundances @ indices 6+
delta_t = scaled_final[:, 1].unsqueeze(dim = 1) # Just time column @ index 1
final_features = scaled_final[:, 6:] # Keep abundances @ indices 6+ 
# Put these into the AbunAfterDt class
all_data = preprocessing.AbunAfterDt(initial_features, delta_t, final_features)

# Do a train-test split
# Set the fraction of data to use for model training
train_frac = 0.7
# Get the size of the training set by rounding
train_length = int(train_frac * len(all_data))
# Use the rest of the data for test set
test_length = len(all_data) - train_length
# Split the data 
# Set up a generator
generator = torch.Generator(device = device)
train_class, test_class = torch.utils.data.random_split(all_data, (train_length, test_length), generator = generator)

# Set up the data in batches
dataloader_train = torch.utils.data.DataLoader(train_class, batch_size = 64, shuffle = True, generator = generator)
dataloader_test = torch.utils.data.DataLoader(test_class, batch_size = 64, shuffle = True, generator = generator)

# Set up the models
# Determine the number of abundance features by removing the non-abundance columns (sample index, time, 4 sampled parameters)
num_feat = len(initial_tensor[0,:]) - 6
print(num_feat) # Just to check
# Create encoder and decoder instances with the appropriate dimensions and pass to the torch device
encoder = architecture.Encoder(num_features = num_feat, latent_dim = latent_dim).to(device)
decoder = architecture.Decoder(num_features = num_feat, latent_dim = latent_dim).to(device)
# Create time emulator
time_emulator = architecture.TimeEmulator(latent_dim = latent_dim, hidden_layer_width = hidden_layer_width, 
                                          num_hidden_layers = num_hidden_layers).to(device)
torch.autograd.set_detect_anomaly(True) # To detect anomalies
# Train the model using function from "model_training" module
train_losses, test_losses = model_training.training(encoder = encoder, decoder = decoder, 
                                                    time_emulator = time_emulator,
                                                    train_batches = dataloader_train, 
                                                    test_batches = dataloader_test, 
                                                    device = device, learning_rate = 0.01, 
                                                    epochs = 10) # Parameters

# Extract value of the train and test loss at each epoch
tr_loss = [x.item() for x in train_losses]
te_loss = [x.item() for x in test_losses]
# Save the values
print('Training loss: ', tr_loss)
print('Test loss: ', te_loss)

# Plot the loss curve
epochs = np.arange(1, 11)
plt.plot(epochs, tr_loss, label = 'Training loss') # Plot training and test loss (vs epoch number)
plt.plot(epochs, te_loss, label = 'Test loss')
plt.yscale('log') # Loss function on a log-scale
plt.xlabel('Epoch') # Label axes
plt.ylabel('Reconstruction loss')
#plt.suptitle('Latent dimension = ' + str(latent_dim)) # Title plot with latent space dimension
plt.legend() # Legend showing loss type
plt.savefig('test_full_pipeline_loss_curve.pdf') # Save the plot
plt.close()