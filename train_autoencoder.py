# Script to train the autoencoder
# Usage: python train_autoencoder.py samples_dir/ latent_dim 

# Import statements
import sys # Command line arguments
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import torch # Torch for ML
import matplotlib.pyplot as plt # Plotting
# Use ML pipeline set up here
import architecture # ML model architecture
import model_training # Training steps
import preprocessing # Feature scaling

# Unpack command line arguments
# Name of this file, path location of data, number of latent space dimensions
(pyfilename, samples_dir, latent_dim) = sys.argv 
# Convert latent_dim to integer
latent_dim = int(latent_dim)

# Torch set up
# Check if GPU is available (use CPU if not)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device") # Print device being used
# Set number of threads used to 8
torch.set_num_threads(8)
# Print the number of CPUs seen by torch to confirm
print('CPUs to be used by torch:', torch.get_num_threads())

# Read in the training
# Create a list to hold all the dataframes
dfs_list = []
# Loop through 1024 training samples
for index in range(1024):
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

# Prepare the data for input to the autoencoder
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
# Convert this dataframe to a torch tensor (with dtype = torch.float32 to be consistent with network weights)
data_tensor = torch.from_numpy(filtered_data.values).to(dtype = torch.float32)
# Rescale the log-abundance values using the function defined in the "preprocessing" module
scaled_data = preprocessing.scale_features(data_tensor)

# Do a train-test split
# Set the fraction of data to use for model training
train_frac = 0.7
# Get the size of the training set by rounding
train_length = int(train_frac * len(all_data))
# Use the rest of the data for test set
test_length = len(all_data) - train_length

# Split the scaled data (keeping sample index, time, and sampled parameter columns)
# Set a random seed
generator = torch.Generator().manual_seed(2583)
train_data, test_data = torch.utils.data.random_split(scaled_data, (train_length, test_length), generator = generator)

# Set up the data in batches
dataloader_train = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

# Set up the models
# Determine the number of abundance features by removing the non-abundance columns (sample index, time, 4 sampled parameters)
num_feat = len(scaled_data[0,:]) - 6
# Create encoder and decoder instances with the appropriate dimensions and pass to the torch device
encoder = architecture.Encoder(num_features = num_feat, latent_dim = latent_dim).to(device)
decoder = architecture.Decoder(num_features = num_feat, latent_dim = latent_dim).to(device)

# Train the model using function from "model_training" module
train_losses, test_losses = model_training.training(encoder = encoder, decoder = decoder, 
                                                    train_batches = dataloader_train, 
                                                    test_batches = dataloader_test, 
                                                    device = device, 
                                                    learning_rate = 0.003, epochs = 100) # Parameters

# Extract value of the train and test loss at each epoch
tr_loss = [x.item() for x in train_losses]
te_loss = [x.item() for x in test_losses]
# Save the values
np.savez('autoencoder_ld_' + str(latent_dim) + '_losses.npz', tr_loss, te_loss)

# Plot the loss curve
epochs = np.arange(1, 101)
plt.plot(epochs, tr_loss, label = 'Training loss') # Plot training and test loss (vs epoch number)
plt.plot(epochs, te_loss, label = 'Test loss')
plt.yscale('log') # Loss function on a log-scale
plt.xlabel('Epoch') # Label axes
plt.ylabel('Reconstruction loss')
plt.suptitle('Latent dimension = ' + str(latent_dim)) # Title plot with latent space dimension
plt.legend() # Legend showing loss type
plt.savefig('autoencoder_ld_' + str(latent_dim) + '_loss_curve.pdf') # Save the plot
plt.close()

# Test looking at the latent space evolution for first chempl training run
first_chempl_run = scaled_data[:462,:] # Get data all 461 time values in first training run
times = first_chempl_run[:, 1].numpy() # Get the times
first_latents = encoder(first_chempl_run[:, 6:].to(device)) # Pass the abundances to the encoder
for i in range(latent_dim): # Loop through features
    # Plot latent space values vs. time for each feature
    plt.plot(times, first_latents[:, i].detach().cpu().numpy(), label = 'Latent feature ' + str(i))
plt.xlabel('Time [yr]') # Label axes
plt.ylabel('Latent feature value')
plt.xscale('log') # Time on a log-scale
plt.suptitle('Latent features for first chempl training run')
plt.legend() # Legend
plt.savefig('autoencoder_ld_' + str(latent_dim) + '_latents.pdf')
plt.close()

# Save the trained models
torch.save(encoder.state_dict(), 'encoder_ld_' + str(latent_dim))
torch.save(decoder.state_dict(), 'decoder_ld_' + str(latent_dim))