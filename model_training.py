# Define functions to train encoder -> time emulator -> decoder pipeline

# Import statements
import torch
from torch import nn
from itertools import chain
import numpy as np

# Define the loss function (as an instance of nn.Module)
class ReconstructionLoss(nn.Module):
    # Initialization
    def __init__(self):
        # Initialize the superclass nn.Module
        super().__init__()
    
    # Forward function (calculates the loss)
    def forward(self, input, target, min_abundance = 1e-20):
        '''
        Inputs:
        input (torch.tensor): Features predicted by the model (after encoder and decoder)
        target (torch.tensor): The true original features
        min_abundance (float): Minimum value for the abundances y_i (defaults to 1e-20). 
        log(y_i) is rescaled from the interval [-log(min_abundance), 0] to [-1, 1]
        Outputs:
        error (float): The mean squared error
        '''

        # Unscale the abundances
        min_log_y = np.log10(min_abundance) # Get minimum value of log10(y_i)
        input_abuns = (min_log_y / 2) - ((min_log_y / 2)  * input) # Unscale inputs
        target_abuns = (min_log_y / 2) - ((min_log_y / 2) * target)

        # Set a small threshold to avoid 0 losses
        epsilon = 1e-8
        # Square the difference between input and target for each feature (adding threshold to avoid nans in backwards pass)
        squared_deviations = torch.pow(input_abuns - target_abuns + epsilon, 2)
        # Get the mean (averaged over the number of features)
        error = torch.mean(squared_deviations)
        # Return the result
        return error
    
# Define function to iterate TimeEmulator (see architecture) over substeps
def iterate_time_emulator_twice(time_emulator, initial_latents, dt):
    '''
    Predict final encoded abundances using the model, iterated through 2 substeps
    Inputs: 
    time_emulator (DeepONet): instance of TimeEmulator function from architecture.py
    initial_latents (torch.tensor): (N * (latent_dim + 4)) tensor containing initial encoded abundances and physical parameters
    dt (torch.tensor): (N * 1) tensor containing timestep values
    Outputs:
    evolved_latents (torch.tensor): (N * latent_dim) tensor containing evolved encoded abundances
    '''

    # Get time ratio for calculating the length of each substep
    ratio = torch.pow(dt, 0.5)
    # Length of step 1 is just the ratio
    step_1_length = ratio
    # Length of step 2 is dt minus step 1
    step_2_length = dt - step_1_length
    # Apply the model once
    evolved_latents_1 = time_emulator((initial_latents, step_1_length))
    # Update the abundance values
    evolved_input = initial_latents.clone()
    evolved_input[:, 4:] = evolved_latents_1
    # Apply the model again
    evolved_latents_2 = time_emulator((evolved_input, step_2_length))
    return evolved_latents_2 # Return last predicted abundance value

# Training process in each epoch
def training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    optimizer (torch.optim): optimizer used to update model weights in backpropagation
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    Outputs:
    avg_loss (float): Average loss over the training batches
    '''
    
    # Set neural networks to training mode
    encoder.train()
    decoder.train()
    time_emulator.train()

    # Set up variable for the total loss
    total_loss = 0

    # Loop through training batches
    for batch, features in enumerate(train_batches):
        # Unpack initial, dt, and final tensors from features (see AbunAfterDt class in preprocessing.py)
        initial = features[0][0]
        dt = features[0][1]
        target = features[1]

        # Extract the physical parameters (columns 0-3) and pass to the torch device
        physical_params = initial[:, :4].to(device)
        # Extract the abundances from the features (columns 4+) and pass to the torch device
        initial_abundances = initial[:, 4:].to(device)
        # Apply encoder to abundances get latent space representation
        initial_latents = encoder(initial_abundances)
        # Concatenate physical parameters (physical_params) and encoded abundances (initial latents)
        initial_for_te = torch.cat((physical_params, initial_latents), dim = 1)
        '''
        # Initialize tensor to pass to time_emulator [batch_size * (latent_dim + 4)]
        initial_for_te = torch.zeros(len(dt), len(initial_latents[0]) + 4, requires_grad = True)
        # First 4 columns are the physical paramaters from initial
        initial_for_te[:, :4] = initial[:, :4].clone()
        # Remaining columns are the encoded abundances
        initial_for_te[:, 4:] = initial_latents.clone()
        '''
        # Apply time emulator iteratively to evolve latents
        evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt.to(device))
        # Apply decoder to get predict evolved abundances
        pred = decoder(evolved_latents)

        # Calculate the loss function
        loss = loss_function(target.to(device), pred)
        # Add this to the total loss
        total_loss += loss

        # Backpropagation
        loss.backward()
        # Apply optimizer to update weights
        optimizer.step()
        # Reset the gradients before the next step
        optimizer.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            print('Loss: %.4f' % loss.item())

    # Get the average loss
    avg_loss = total_loss / len(train_batches)
    # Return this
    return avg_loss

        

# Testing process in each epoch
def testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device): 
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    test_batches (torch.tensor): tensor containing features for testing data (grouped in batches)
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    Outputs:
    avg_loss (float): Average loss over the test batches
    '''

    # Set all neural networks to evaluation mode
    encoder.eval()
    decoder.eval()
    time_emulator.eval()

    # Sum variable to add up test loss for each batch
    total_loss = 0

    # Loop through test batches
    # WITHOUT calculating gradients
    with torch.no_grad():
        for batch, features in enumerate(test_batches):
            # Unpack initial, dt, and final tensors from features (see AbunAfterDt class in preprocessing.py)
            initial = features[0][0]
            dt = features[0][1]
            target = features[1]            

            # Extract the physical parameters (columns 0-3) and pass to the torch device
            physical_params = initial[:, :4].to(device)
            # Extract the abundances from the input features (columns 6+) and pass to torch device
            initial_abundances = initial[:, 4:].to(device)
            # Apply encoder to get latent space representation
            initial_latents = encoder(initial_abundances)
            # Concatenate physical parameters (physical_params) and encoded abundances (initial latents)
            initial_for_te = torch.cat((physical_params, initial_latents), dim = 1)
            '''
            # Initialize tensor to pass to time_emulator [batch_size * (latent_dim + 4)]
            initial_for_te = torch.zeros(len(dt), len(initial_latents[0]) + 4)
            # First 4 columns are the physical paramaters from initial
            initial_for_te[:, :4] = initial[:, :4]
            # Remaining columns are the encoded abundances
            initial_for_te[:, 4:] = initial_latents
            '''
            # Iterate time emulator to evolve latent space features
            evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt.to(device))
            # Apply decoder to get predicted evolved abundances
            pred = decoder(evolved_latents)
            # Calculate loss in this batch
            batch_loss = loss_function(target.to(device), pred)
            # Add this to the total loss
            total_loss += batch_loss

    # Calculate average loss
    avg_loss = total_loss / len(test_batches)
    # Print this
    print('Average test loss: %.4f' % avg_loss)
    # Return the average loss
    return avg_loss

# Define entire training procedure
def training(encoder, decoder, time_emulator, train_batches, test_batches, device,
             learning_rate = 1e-3, epochs = 100):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    test_batches (torch.tensor): tensor containing features for test data (grouped in batches)
    device (str): "cpu" or torch.accelerator.current_accelerator()
    learning_rate (float): scalar multiplying gradient in weight updates
    epochs (int): number of train-test cycles (each uses all train and test batches)
    Outputs:
    train_losses (list of floats): List of the average training loss for each epoch
    test_losses (list of floats): List of the average test loss for each epoch
    '''

    # Set up ADAM optimizer to update encoder and decoder parameters
    optimizer = torch.optim.Adam(params = chain(encoder.parameters(), time_emulator.parameters(), decoder.parameters()),
                                 lr = learning_rate)
    
    # Set up the loss function as the reconstruction loss defined above
    loss_function = ReconstructionLoss()

    # Set up arrays for the training and test loss 
    train_losses = []
    test_losses = []

    # Loop through epochs
    for epoch in range(epochs):
        # Print epoch number to track progress
        print('Epoch: %i' % (epoch + 1))
        # Training step
        train_loss = training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device)
        train_losses.append(train_loss) # Add this to the appropriate array
        # Testing step
        test_loss = testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device)
        test_losses.append(test_loss) # Add this to the appropriate array

    # Return the train and test loss arrays
    return train_losses, test_losses
    print('Done!')