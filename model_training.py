# Define function to train encoder and decoder

# Import statements
import torch
from torch import nn
from itertools import chain

# Define the loss function (as an instance of nn.Module)
class ReconstructionLoss(nn.Module):
    # Initialization
    def __init__(self):
        # Initialize the superclass nn.Module
        super().__init__()
    
    # Forward function (calculates the loss)
    def forward(self, input, target):
        '''
        Inputs:
        input (torch.tensor): Features predicted by the model (after encoder and decoder)
        target (torch.tensor): The true original features
        Outputs:
        error (float): The mean squared error
        '''

        # Square the difference between input and target for each feature
        squared_deviations = torch.pow(input - target, 2)
        # Get the mean (averaged over the number of features)
        error = torch.mean(squared_deviations)
        # Return the result
        return error

# Training process in each epoch
def training_step(encoder, decoder, train_batches, optimizer, loss_function, device):
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
    
    # Set both models to training mode
    encoder.train()
    decoder.train()

    # Set up variable for the total loss
    total_loss = 0

    # Loop through training batches
    for batch, features in enumerate(train_batches):
        # Extract the abundances from the features (columns 6+) and pass to the torch device
        abundances = features[:, 6:].to(device)
        # Apply encoder to abundances get latent space representation
        latents = encoder(abundances)
        # Apply decoder to get reconstruction in feature space
        reconstruction = decoder(latents)

        # Calculate the loss function
        loss = loss_function(abundances, reconstruction)
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
def testing_step(encoder, decoder, test_batches, loss_function, device): 
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

    # Set encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()

    # Get number of batches (to calculate average loss per batch)
    num_batches = len(test_batches)
    # Sum variable to add up test loss for each batch
    total_loss = 0

    # Loop through test batches
    # WITHOUT calculating gradients
    with torch.no_grad():
        for batch, features in enumerate(test_batches):
            # Extract the abundances from the input features (columns 6+) and pass to torch device
            abundances = features[:, 6:].to(device)
            # Apply encoder to get latent space representation
            latents = encoder(abundances)
            # Apply decoder to get reconstruction in feature space
            reconstruction = decoder(latents)
            # Calculate loss in this batch
            batch_loss = loss_function(abundances, reconstruction)
            # Add this to the total loss
            total_loss += batch_loss

    # Calculate average loss
    avg_loss = total_loss / num_batches
    # Print this
    print('Average test loss: %.4f' % avg_loss)
    # Return the average loss
    return avg_loss

def training(encoder, decoder, train_batches, test_batches, device,
             learning_rate = 1e-3, epochs = 100):
    '''
    Inputs:
    encoder (Encoder): instance of Encoder class defined in architecture.py
    decoder (Decoder): instance of Decoder class defined in architecture.py
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
    optimizer = torch.optim.Adam(params = chain(encoder.parameters(), decoder.parameters()),
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
        train_loss = training_step(encoder, decoder, train_batches, optimizer, loss_function, device)
        train_losses.append(train_loss) # Add this to the appropriate array
        # Testing step
        test_loss = testing_step(encoder, decoder, test_batches, loss_function, device)
        test_losses.append(test_loss) # Add this to the appropriate array

    # Return the train and test loss arrays
    return train_losses, test_losses
    print('Done!')