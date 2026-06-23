# Define the forward pass of the neural networks making up the emulator

# Import statements
import torch # pytorch for neural networks
from torch import nn # Specifically import neural network module
import deepxde as dde # DeepXDE for DeepONet architecture used in time emulator

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
            nn.Linear(in_features = num_features, out_features = 128), # num_features -> 128, linear layer
            nn.Tanh(), # Hyperbolic tangent activation function
            nn.Linear(in_features = 128, out_features = 32), # 128 -> 32, linear
            nn.Tanh(),
            nn.Linear(in_features = 32, out_features = latent_dim), # 32 -> latent_dim, linear
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
            nn.Linear(in_features = latent_dim, out_features = 32),
            nn.Tanh(),
            nn.Linear(in_features = 32, out_features = 128),
            nn.Tanh(),
            nn.Linear(in_features = 128, out_features = num_features),
            nn.Tanh()
        )

    def forward(self, latents):
        latents = self.flatten(latents)
        outputs = self.operations(latents)
        return outputs
    
# Define the DeepONet time emulator
def TimeEmulator(latent_dim, hidden_layer_width, num_hidden_layers):
    '''
    Function to define the DeepONet time emulator
    Inputs:
    latent_dim (int): Number of latent dimensions from the autoencoder
    hidden_layer_width (int): Number of nodes per hidden layer in the DeepONet branch/trunk networks
    num_hidden_layers (int): Number of hidden layers in the DeepOnet branch/trunk networks
    Outputs:
    deeponet_model (nn.Module): The DeepONet neural network with the specified architecture
    '''

    # Network shape parameters
    branch_inputs = latent_dim + 4 # Branch network takes in autoencoder latent features + 4 sampled physical parameters
    trunk_inputs = 1 # Trunk network takes in delta_t
    num_outputs = latent_dim # DeepONet outputs latent features (evolved over timestep delta_t)
    # Define network shape with needed inputs, outputs, hidden layer number of width from function inputs
    branch_shape = [branch_inputs] + num_hidden_layers * [hidden_layer_width] + [num_outputs]
    # use DeepONet mode where trunk output is split for each total network output
    # so need (num_output ^ 2) nodes in final layer of trunk network
    trunk_shape = [trunk_inputs] + num_hidden_layers * [hidden_layer_width] * [num_outputs ** 2]

    # Configure DeepONet
    deeponet_model = dde.nn.pytorch.deeponet.DeepONet(layer_sizes_branch = branch_shape, # Branch network shape defined above
                                                      layer_sizes_trunk = trunk_shape, # Trunk network shape defined above
                                                      activation = 'Tanh', # Features all in [-1, 1] interval
                                                      kernel_initializer = 'Glorot normal',
                                                      num_outputs = num_outputs, # Define number of outputs
                                                      multi_output_strategy = 'split_trunk' # Split trunk network for each output
    )
    return deeponet_model
