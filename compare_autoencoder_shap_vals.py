# Script to compare autoencoder SHAP values for different latent dimensions
# Usage: python compare_autoencoder_shap_vals.py samples_dir/ latent_dim 

# Import statements
import sys # sys for command line arguments
import pandas as pd # pandas for dataframe handling
import numpy as np # numpy for math
import torch # torch for tensors, neural network architecture
import shap # shap for feature importance
import matplotlib.pyplot as plt # matplotlib for plotting
# Use training pipeline modules
import architecture
import model_training
import preprocessing

# Set number of threads used to 8
torch.set_num_threads(8)
# Print the number of CPUs seen by torch to confirm
print('CPUs to be used by torch:', torch.get_num_threads())

# Unpack command line arguments
# Name of this file, path location of data, number of latent space dimensions
(pyfilename, samples_dir, latent_dim) = sys.argv 
# Convert latent_dim to integer
latent_dim = int(latent_dim)

# Read in the chempl data
# Create a list to hold all the dataframes
dfs_list = []
# Loop through training samples
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

# Look at where the data is under some threshold
min_abundance = 1e-20
# Only keep abundance columns with some values above this threshold
nonzero_abuns = all_data[all_data.columns[7:]].loc[:, (all_data > min_abundance).any()]
# Check if each abundance is below some threshold, and set abundance to the threshold if so
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
# Scale it using the function from the preprocessing module
scaled_data = preprocessing.scale_features(data_tensor)

# Now select a random sample of 100 points to pass to shap
# Create a numpy random number generator with a specific speed
rng = np.random.default_rng(3865)
# Get 100 random integers within length of array
random_indices = rng.integers(0, len(scaled_data), 100)
# Get rows of scaled_data at these indices
random_sample = scaled_data[random_indices]

# Get list of abundance column names (to associate feature indices with species names)
species_names = filtered_data.columns[6:]

# Set up the trained autoencoder models
# Get number of features (number of non-trivial abundance columns in the data)
num_feat = len(scaled_data[0,:]) - 6
# Create encoder and decoder instances with the appropriate dimensions
encoder = architecture.Encoder(num_features = num_feat, latent_dim = latent_dim)
decoder = architecture.Decoder(num_features = num_feat, latent_dim = latent_dim)
# Read in the trained models
encoder.load_state_dict(torch.load('encoder_ld_' + str(latent_dim), weights_only=True))
decoder.load_state_dict(torch.load('decoder_ld_' + str(latent_dim), weights_only=True))

# Apply the model to the random sample of 100 points
first_latents = encoder(random_sample[:, 6:]) # Apply the trained encoder to abudance columns

# Create shap explainers for the encoder and decoder, averaged over the random sample
exp_enc = shap.DeepExplainer(model = encoder, data = random_sample[:, 6:]) # Encoder
exp_dec = shap.DeepExplainer(model = decoder, data = first_latents) # Decoder
# Calculate SHAP values from these explainers
shap_values_enc = exp_enc.shap_values(random_sample[:, 6:]) # Encoder
shap_values_dec = exp_dec.shap_values(first_latents[:20]) # Decoder 
# Note: currently only use first 20 random samples to calculate decoder SHAP values
# because decoder SHAP values are slow to calculate since the decoder has 573 outputs

# Print, visualize, and save results
# Encoder
# Creae a bar plot with the top 10 most important species
sp_enc = shap.summary_plot(shap_values_enc, plot_type = 'bar', feature_names = species_names, max_display = 10)
# Save it
plt.savefig('autoencoder_ld_' + str(latent_dim) + '_enc_shap.pdf')
# For each input abundance, loop through all latent features and add up the mean |SHAP|
total_shap_dict = {}
for abundance in range(num_feat): # Loop through input abundances (x573)
    # Counter variable for total mean |SHAP|
    total_mean_shap = 0
    for latent_feat in range(latent_dim): # Loop through latent features (x30)
        # Get mean |SHAP| value and add to running total
        total_mean_shap += np.mean(np.abs(shap_values_enc[:, abundance, latent_feat]))
    # Get species name
    species = species_names[abundance]
    # Put species name, total mean |SHAP| in a dictionary
    total_shap_dict[species] = float(total_mean_shap)
# Sort the values by mean |SHAP| in descending order
mean_shaps = list(total_shap_dict.values()) # Get list of mean |SHAP|
indices_by_desc_shap = np.argsort(mean_shaps)[::-1] # Get species indices by descending mean |SHAP|
sorted_shap_dict = {species_names[i] : mean_shaps[i] for i in indices_by_desc_shap} # Put species, mean |SHAP| pairs in a new dict
# Convert the dictionary of species and mean |SHAP| to a pandas Series and save it
sorted_shap_df = pd.Series(sorted_shap_dict) # Convert to a pandas series
sorted_shap_df.to_string(buf = 'autoencoder_ld_' + str(latent_dim) + '_enc_shap.dat', header = False)

# Decoder
# Set threshold for interesting mean |SHAP|
thresh = 0.2
# Open a file to write values
f = open('autoencoder_ld_' + str(latent_dim) + '_dec_shap.dat', 'w')
# Iterate through features
for abundance in range(num_feat): # Loop through input abundances (x573)
    species = species_names[abundance] # Get species name
    latent_shaps = np.zeros(latent_dim)
    for latent in range(latent_dim): # Loop through latent dimensions
        latent_shaps[latent] = np.mean(np.abs(shap_values_dec[:, latent, abundance])) # Get mean |SHAP|
        if latent_shaps[latent] > thresh: # If mean |SHAP| is above threshold value, write it to the file
            f.write(species + ' ' + str(latent) + ' ' + str(latent_shaps[latent]) + '\n') # Write the results
f.close() # Close the file