# Define functions to train encoder -> time emulator -> decoder pipeline

# Import statements
import torch
from torch import nn
from itertools import chain
import numpy as np
import gc
import preprocessing # Need dt scaling function from preprocessing module

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
        #input_abuns = (min_log_y / 2) - ((min_log_y / 2)  * input) # Unscale inputs
        #target_abuns = (min_log_y / 2) - ((min_log_y / 2) * target)

        # Set a small threshold to avoid 0 losses
        epsilon = 1e-8
        # Square the difference between input and target for each feature (adding threshold to avoid nans in backwards pass)
        # multiply by (min_log_y / 2)^2 to convert back to physical space
        squared_deviations = torch.pow(input - target+ epsilon, 2) * ((min_log_y / 2) ** 2)
        # Get the mean (averaged over the number of features)
        error = torch.mean(squared_deviations)
        # Return the result
        return error
    
# Define function to iterate TimeEmulator (see architecture) over substeps
def iterate_time_emulator_twice(time_emulator, initial_latents, dt, device, dt_min = 1e1, dt_max = 1e8):
    '''
    Predict final encoded abundances using the model, iterated through 2 substeps
    Inputs: 
    time_emulator (DeepONet): instance of TimeEmulator function from architecture.py
    initial_latents (torch.tensor): (N * (latent_dim + 4)) tensor containing initial encoded abundances and physical parameters
    device (str): "cpu" or torch.accelerator.current_accelerator()
    dt (torch.tensor): (N * 1) tensor containing timestep values
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
    Outputs:
    evolved_latents (torch.tensor): (N * latent_dim) tensor containing evolved encoded abundances
    '''
    '''
    # Get time ratio for calculating the length of each substep
    ratio = torch.pow(dt, 0.5)
    # Length of step 1 is just the ratio
    step_1_length = ratio
    # Rescale step 1 (need to put it on cpu first)
    scaled_step_1 = preprocessing.scale_dt(step_1_length, dt_min = dt_min, dt_max = dt_max).to(device)
    # Length of step 2 is dt minus step 1
    step_2_length = dt - step_1_length
    # Rescale step 2 (need to put it on cpu first)
    scaled_step_2 = preprocessing.scale_dt(step_2_length, dt_min = dt_min, dt_max = dt_max).to(device)
    '''
    # Get length of half step
    half_step_length = dt * 0.5
    # Rescale step length
    scaled_half_step = preprocessing.scale_dt(half_step_length, dt_min = dt_min, dt_max = dt_max).to(device)
    # Apply the model once
    evolved_latents_1 = time_emulator((initial_latents, scaled_half_step))
    # Update the abundance values
    evolved_input = initial_latents.clone()
    evolved_input[:, 4:] = evolved_latents_1
    # Apply the model again
    evolved_latents_2 = time_emulator((evolved_input, scaled_half_step))

    # Delete created tensors
    # del ratio, step_1_length, scaled_step_1, step_2_length, scaled_step_2, evolved_latents_1, evolved_input
    del half_step_length, scaled_half_step, evolved_latents_1, evolved_input
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return evolved_latents_2 # Return last predicted abundance value

# Training process in each epoch
def training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device, dt_min = 1e1, dt_max = 1e8):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    optimizer (torch.optim): optimizer used to update model weights in backpropagation
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
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
        # Apply time emulator iteratively to evolve latents
        evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt, device = device, dt_min = dt_min, dt_max = dt_max)
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

        # Remove the loaded tensors
        del initial, dt, target, physical_params, initial_abundances, initial_latents, initial_for_te, evolved_latents, pred
        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Print loss every 100 batches
        if batch % 100 == 0:
            print('Loss: %.4f' % loss.item())

    # Get the average loss
    avg_loss = total_loss / len(train_batches)
    # Return this
    return avg_loss

# Testing process in each epoch
def testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device, dt_min = 1e1, dt_max = 1e8): 
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    test_batches (torch.tensor): tensor containing features for testing data (grouped in batches)
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
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
            # Apply time emulator twice
            evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt, device = device, dt_min = dt_min, dt_max = dt_max)
            '''
            # Apply time emulator ONCE (to check that training step doesn't learn 
            # something weird that works for two iterations but not a single application)
            scaled_timestep = preprocessing.scale_dt(dt, dt_min = dt_min, dt_max = dt_max).to(device) # Scale timestep
            evolved_latents = time_emulator((initial_for_te.to(device), scaled_timestep)) # Apply time_emulator
            '''
            # Apply decoder to get predicted evolved abundances
            pred = decoder(evolved_latents)
            # Calculate loss in this batch
            batch_loss = loss_function(target.to(device), pred)
            # Add this to the total loss
            total_loss += batch_loss

            # Remove the loaded tensors
            del initial, dt, target, physical_params, initial_abundances, initial_latents, initial_for_te, evolved_latents, pred
            # Clear the cache
            gc.collect()
            torch.cuda.empty_cache()

    # Calculate average loss
    avg_loss = total_loss / len(test_batches)
    # Print this
    print('Average test loss: %.4f' % avg_loss)
    # Return the average loss
    return avg_loss

# Define entire training procedure
def training_loop(encoder, decoder, time_emulator, train_batches, test_batches, device,
                  learning_rate = 0.03, epochs = 100, dt_min = 1e1, dt_max = 1e8, to_save = True):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    test_batches (torch.tensor): tensor containing features for test data (grouped in batches)
    device (str): "cpu" or torch.accelerator.current_accelerator()
    learning_rate (float): initial scalar multiplying gradient in weight updates
    epochs (int): number of train-test cycles (each uses all train and test batches)
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
    to_save (bool): whether or not to save the best-performing model
    Outputs:
    train_losses (list of floats): List of the average training loss for each epoch
    test_losses (list of floats): List of the average test loss for each epoch
    '''

    # Set up ADAM optimizer to update encoder and decoder parameters
    optimizer = torch.optim.Adam(params = chain(encoder.parameters(), time_emulator.parameters(), decoder.parameters()),
                                 lr = learning_rate, fused = True)
    # Set up a learning rate scheduler to decay learning rate by 5% every epoch
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    
    # Set up the loss function as the reconstruction loss defined above
    loss_function = ReconstructionLoss()

    # Set up arrays for the training and test loss 
    train_losses = []
    test_losses = []

    # Variable to hold best test set loss
    best_test_loss = np.inf

    # Loop through epochs
    for epoch in range(epochs):
        # Print epoch number to track progress
        print('Epoch: %i' % (epoch + 1))
        # Training step
        train_loss = training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device, dt_min = dt_min, dt_max = dt_max)
        train_losses.append(train_loss) # Add this to the appropriate array
        # Testing step
        test_loss = testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device, dt_min = dt_min, dt_max = dt_max)
        test_losses.append(test_loss) # Add this to the appropriate array
        # Step the learning rate scheduler
        # scheduler.step()

        # Check if current test loss improves on stored best value
        if test_loss < best_test_loss:
            # If so, update the best loss value
            best_test_loss = test_loss
            # and store current models
            best_encoder = encoder
            best_time_emulator = time_emulator
            best_decoder = decoder

        # Clear the cache
        torch.cuda.empty_cache()

    # Save the models from epoch with best test loss if to_save flag is True
    if to_save == True:
        torch.save(best_encoder.state_dict(), 'encoder')
        torch.save(best_time_emulator.state_dict(), 'time_emulator')
        torch.save(best_decoder, 'decoder')
    # Set each model to the best value
    encoder = best_encoder
    time_emulator = best_time_emulator
    decoder = best_decoder

    # Return the train and test loss arrays
    return train_losses, test_losses
    print('Done!')

# Define training step for just the autoencoder
def ae_training_step(encoder, decoder, train_batches, optimizer, loss_function, device):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    optimizer (torch.optim): optimizer used to update model weights in backpropagation
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    Outputs:
    avg_loss (float): Average loss over the training batches
    '''

    # Set encoder and decoder to training mode
    encoder.train()
    decoder.train()

    # Set up variable for the total loss
    total_loss = 0

    # Loop through training batches
    for batch, features in enumerate(train_batches):
        # Unpack initial, dt, and final tensors from features (see AbunAfterDt class in preprocessing.py)
        initial = features[0][0]
        target = features[1]

        # Extract the (scaled) abundances in columns 4+ of initial 
        initial_abundances = initial[:, 4:]
        # Concatenate these (along axis 0) with the target abundances (so autoencoder training sees both initial and final states)
        all_abundances = torch.cat((initial_abundances, target), dim = 0).to(device)

        # Apply emulator to get latent features
        latents = encoder(all_abundances)
        # Apply decoder to get reconstruction of (scaled) abundances
        reconstruction = decoder(latents)

        # Calculate loss
        loss = loss_function(all_abundances, reconstruction)
        # Add this to the total loss
        total_loss += loss

        # Backpropagation
        loss.backward()
        # Apply optimizer to update weights
        optimizer.step()
        # Reset the gradients before the next step
        optimizer.zero_grad()

        # Remove the loaded tensors
        del initial, target, initial_abundances, all_abundances, latents, reconstruction
        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Print loss every 100 batches
        if batch % 100 == 0:
            print('Loss: %.4f' % loss.item())

    # Get the average loss
    avg_loss = total_loss / len(train_batches)
    # Return this
    return avg_loss

def ae_testing_step(encoder, decoder, test_batches, loss_function, device): 
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    test_batches (torch.tensor): tensor containing features for testing data (grouped in batches)
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    Outputs:
    avg_loss (float): Average loss over the test batches
    '''

    # Set all neural networks to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Sum variable to add up test loss for each batch
    total_loss = 0
    
    # Loop through test batches
    # WITHOUT calculating gradients
    with torch.no_grad():
        for batch, features in enumerate(test_batches):
            # Unpack initial, dt, and final tensors from features (see AbunAfterDt class in preprocessing.py)
            initial = features[0][0]
            target = features[1]            
    
            # Extract the abundances from the input features (columns 6+) and pass to torch device
            initial_abundances = initial[:, 4:]
            # Concatenate these (along axis 0) with the target abundances (so autoencoder training sees both initial and final states)
            all_abundances = torch.cat((initial_abundances, target), dim = 0).to(device)

            # Apply encoder to get latent space representation
            latents = encoder(all_abundances)
            # Apply decoder to get predicted evolved abundances
            reconstruction = decoder(latents)

            # Calculate loss in this batch
            batch_loss = loss_function(all_abundances, reconstruction)
            # Add this to the total loss
            total_loss += batch_loss
    
            # Remove the loaded tensors
            del initial, target, initial_abundances, all_abundances, latents, reconstruction
            # Clear the cache
            gc.collect()
            torch.cuda.empty_cache()
    
    # Calculate average loss
    avg_loss = total_loss / len(test_batches)
    # Print this
    print('Average test loss: %.4f' % avg_loss)
    # Return the average loss
    return avg_loss

def ae_training_loop(encoder, decoder, train_batches, test_batches, device,
                     learning_rate = 0.03, epochs = 100, to_save = True):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    test_batches (torch.tensor): tensor containing features for test data (grouped in batches)
    device (str): "cpu" or torch.accelerator.current_accelerator()
    learning_rate (float): initial scalar multiplying gradient in weight updates
    epochs (int): number of train-test cycles (each uses all train and test batches)
    to_save (bool): whether or not to save best-performing models
    Outputs:
    train_losses (list of floats): List of the average training loss for each epoch
    test_losses (list of floats): List of the average test loss for each epoch
    '''

    # Set up ADAM optimizer to update encoder and decoder parameters
    optimizer = torch.optim.Adam(params = chain(encoder.parameters(), decoder.parameters()),
                                 lr = learning_rate)
    
    # Set up the loss function as the reconstruction loss defined above
    loss_function = ReconstructionLoss()

    # Set up arrays for the training and test loss 
    train_losses = []
    test_losses = []

    # Variable to hold current best test loss
    best_test_loss = np.inf

    # Loop through epochs
    for epoch in range(epochs):
        # Print epoch number to track progress
        print('Epoch: %i' % (epoch + 1))
        # Training step
        train_loss = ae_training_step(encoder, decoder, train_batches, optimizer, loss_function, device)
        train_losses.append(train_loss) # Add this to the appropriate array
        # Testing step
        test_loss = ae_testing_step(encoder, decoder, test_batches, loss_function, device)
        test_losses.append(test_loss) # Add this to the appropriate array

        # Check if current test loss is better than previous best value
        if test_loss < best_test_loss:
            # If so, make this the new best test loss
            best_test_loss = test_loss
            # Store the encoder and decoder
            best_encoder = encoder
            best_decoder = decoder

        # Clear the cache
        torch.cuda.empty_cache()

    # Save the models from epoch with best test loss if to_save flag is True
    if to_save == True:
        torch.save(best_encoder.state_dict(), 'encoder')
        torch.save(best_decoder.state_dict(), 'decoder')  
    # Set each model to the best version
    encoder = best_encoder
    decoder = best_decoder  

    # Return the train and test loss arrays
    return train_losses, test_losses
    print('Done!')

def time_emulator_training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device, dt_min = 1e1, dt_max = 1e8):
    '''
    Inputs:
    encoder (Encoder): trained instance of Encoder class defined in architecture.py
    decoder (Decoder): trained instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    optimizer (torch.optim): optimizer used to update model weights in backpropagation
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
    Outputs:
    avg_loss (float): Average loss over the training batches
    '''
    
    # Set time emulator network to training mode
    time_emulator.train()
    # Set autoencoders to evaluation mode
    encoder.eval()
    decoder.eval()

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
        # Apply time emulator iteratively to evolve latents (here we need gradients)
        evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt, device = device, dt_min = dt_min, dt_max = dt_max)
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

        # Remove the loaded tensors
        del initial, dt, target, physical_params, initial_abundances, initial_latents, initial_for_te, evolved_latents, pred
        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Print loss every 100 batches
        if batch % 100 == 0:
            print('Loss: %.4f' % loss.item())

    # Get the average loss
    avg_loss = total_loss / len(train_batches)
    # Return this
    return avg_loss

def time_emulator_testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device, dt_min = 1e1, dt_max = 1e8): 
    '''
    Inputs:
    encoder (Encoder): trained instance of Encoder class defined in architecture.py
    decoder (Decoder): trained instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    test_batches (torch.tensor): tensor containing features for testing data (grouped in batches)
    loss_function (torch.nn.Module): loss function comparing features and their autoencoder reconstruction
    device (str): "cpu" or torch.accelerator.current_accelerator()
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
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
            # Apply time emulator twice
            evolved_latents = iterate_time_emulator_twice(time_emulator, initial_for_te.to(device), dt, device = device, dt_min = dt_min, dt_max = dt_max)
            '''
            # Apply time emulator ONCE (to check that training step doesn't learn 
            # something weird that works for two iterations but not a single application)
            scaled_timestep = preprocessing.scale_dt(dt, dt_min = dt_min, dt_max = dt_max).to(device) # Scale timestep
            evolved_latents = time_emulator((initial_for_te.to(device), scaled_timestep)) # Apply time_emulator
            '''
            # Apply decoder to get predicted evolved abundances
            pred = decoder(evolved_latents)
            # Calculate loss in this batch
            batch_loss = loss_function(target.to(device), pred)
            # Add this to the total loss
            total_loss += batch_loss

            # Remove the loaded tensors
            del initial, dt, target, physical_params, initial_abundances, initial_latents, initial_for_te, evolved_latents, pred
            # Clear the cache
            gc.collect()
            torch.cuda.empty_cache()

    # Calculate average loss
    avg_loss = total_loss / len(test_batches)
    # Print this
    print('Average test loss: %.4f' % avg_loss)
    # Return the average loss
    return avg_loss

def time_emulator_training_loop(encoder, decoder, time_emulator, train_batches, test_batches, device,
                                learning_rate = 0.03, epochs = 100, dt_min = 1e1, dt_max = 1e8, to_save = True):
    '''
    Inputs:
    encoder (Encoder): trained instance of Encoder class defined in architecture.py
    decoder (Decoder): trained instance of Decoder class defined in architecture.py
    time_emulator (DeepONet): instance of TimeEmulator defined in architecture.py
    train_batches (torch.tensor): tensor containing features for training data (grouped in batches)
    test_batches (torch.tensor): tensor containing features for test data (grouped in batches)
    device (str): "cpu" or torch.accelerator.current_accelerator()
    learning_rate (float): initial scalar multiplying gradient in weight updates
    epochs (int): number of train-test cycles (each uses all train and test batches)
    dt_min (float): minimum timestep in years (defaults to 10)
    dt_max (float): maximum timestep in years (defaults to 1e8)
    to_save (bool): whether or not to save best performing model
    Outputs:
    train_losses (list of floats): List of the average training loss for each epoch
    test_losses (list of floats): List of the average test loss for each epoch
    '''

    # Set up ADAM optimizer to update encoder and decoder parameters
    optimizer = torch.optim.Adam(params = chain(time_emulator.parameters()),
                                 lr = learning_rate, fused = True)
    # Set up a learning rate scheduler to decay learning rate by 5% every epoch
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    
    # Set up the loss function as the reconstruction loss defined above
    loss_function = ReconstructionLoss()

    # Set up arrays for the training and test loss 
    train_losses = []
    test_losses = []

    # Set up a variable to hold the best test loss
    best_test_loss = np.inf

    # Loop through epochs
    for epoch in range(epochs):
        # Print epoch number to track progress
        print('Epoch: %i' % (epoch + 1))
        # Training step
        train_loss = time_emulator_training_step(encoder, decoder, time_emulator, train_batches, optimizer, loss_function, device, dt_min = dt_min, dt_max = dt_max)
        train_losses.append(train_loss) # Add this to the appropriate array
        # Testing step
        test_loss = time_emulator_testing_step(encoder, decoder, time_emulator, test_batches, loss_function, device, dt_min = dt_min, dt_max = dt_max)
        test_losses.append(test_loss) # Add this to the appropriate array
        # Step the learning rate scheduler
        # scheduler.step()

        # Check if test loss is best so far
        if test_loss < best_test_loss:
            # If so, make this the new best loss
            best_test_loss = test_loss
            # Store this model as the best model
            best_time_emulator = time_emulator

        # Clear the cache
        torch.cuda.empty_cache()

    # Save the models from epoch with best test loss if to_save flag is True
    if to_save == True:
        torch.save(best_time_emulator.state_dict(), 'time_emulator')
    time_emulator = best_time_emulator

    # Return the train and test loss arrays
    return train_losses, test_losses
    print('Done!')