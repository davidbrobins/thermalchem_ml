# Define the forward pass of the neural networks making up the emulator

# Import statements
import torch
from torch import nn

# Define the encoder class (inherits from nn.Module)
class Encoder(nn.Module):
    # Initialization
    def __init__(self, num_features, latent_dim):
        '''
        Inputs:
        num_features (int): Number of features (number of inputs to the encoder)
        latent_dim (int): Dimensionality of the latent space (number of outputs of the encoder)
        '''
        # Initialize the superclass nn.Module
        super().__init__()
        # Define 'flatten' attribute as tensor flattening
        self.flatten = nn.Flatten()
        # Define the neural network operations
        self.operations = nn.Sequential( # Wrap the sequence of linear layers and activation functions
            nn.Linear(in_features = num_features, out_features = 32), # num_features -> 32, linear layer
            nn.Tanh(), # Hyperbolic tangent activation function
            nn.Linear(in_features = 32, out_features = 16), # 32 -> 16, linear
            nn.Tanh(),
            nn.Linear(in_features = 16, out_features = latent_dim), # 16 -> latent_dim, linear
            nn.Tanh()
        )

    # Define the forward mapping of the encoder on a tensor of features 'inputs'
    def forward(self, inputs):
        # Flatten inputs
        inputs = self.flatten(inputs)
        # Apply the encoder to get the latent space tensor
        latents = self.operations(inputs)
        return latents
        

# Analogously define the decoder
class Decoder(nn.Module):
    def __init__(self, num_features, latent_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        # Structure is the encoder linear layers in reverse
        self.operations = nn.Sequential(
            nn.Linear(in_features = latent_dim, out_features = 16),
            nn.Tanh(),
            nn.Linear(in_features = 16, out_features = 32),
            nn.Tanh(),
            nn.Linear(in_features = 32, out_features = num_features),
            nn.Tanh()
        )

    def forward(self, latents):
        latents = self.flatten(latents)
        outputs = self.operations(latents)
        return outputs
