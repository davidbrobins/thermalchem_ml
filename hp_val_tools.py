# Module containing function for optimizing hyperparameters

# Import statements
import torch # pytorch for ML
from sklearn.model_selection import KFold # Function to split data into 'folds' for cross-validation
import architecture # architecture module defining neural network structures
import model_training # model_training module defining training loops for ML models

# Define objective function with 5-vold cross validation
def hp_val_loss(config, data = None, device = None):
    generator = torch.Generator(device = 'cpu') # Just a check
    # Set up models 
    # Get number of features
    num_feat = 573
    # Create encoder and decoder instances with the appropriate dimensions and pass to the torch device
    encoder = architecture.Encoder(num_features = num_feat, latent_dim = config['latent_dim']).to(device)
    decoder = architecture.Decoder(num_features = num_feat, latent_dim = config['latent_dim']).to(device)
    # Create time emulator
    time_emulator = architecture.TimeEmulator(latent_dim = config['latent_dim'], 
                                              hidden_layer_width = config['hidden_layer_width'], 
                                              num_hidden_layers = config['num_hidden_layers']).to(device)

    # Split the validation data into 5 folds for cross-validation
    num_folds = 5
    kf = KFold(n_splits = num_folds, shuffle = True)

    # Loop through k-folds
    total_eval_loss = 0 # Counter to hold total time emulator evaluation loss
    for fold_index, (opt_indices, eval_indices) in enumerate(kf.split(data)):
        print('Fold index: ', fold_index)
        # Get data at right indices
        eval_class = torch.utils.data.Subset(data, eval_indices) # Data to evaluate model
        opt_class = torch.utils.data.Subset(data, opt_indices) # Data to optimize (i.e. train) model
        # Load data in batches
        dataloader_eval = torch.utils.data.DataLoader(eval_class, batch_size = 256, shuffle = True, generator = generator)
        dataloader_opt = torch.utils.data.DataLoader(opt_class, batch_size = 256, shuffle = True, generator = generator)

        # Train autoencoder
        print('Training autoencoder...')
        ae_opt_losses, ae_eval_losses = model_training.ae_training_loop(encoder = encoder, 
                                                                        decoder = decoder,
                                                                        train_batches = dataloader_opt, 
                                                                        test_batches = dataloader_eval,
                                                                        device = device, 
                                                                        learning_rate = config['learning_rate'],
                                                                        epochs = 5, # Shorter for hyperaprameter validation
                                                                        to_save = False) # Do not save models
        print('Training time emulator...')
        # Train time emulator
        te_opt_losses, te_eval_losses = model_training.time_emulator_training_loop(encoder = encoder, 
                                                                                   decoder = decoder, 
                                                                                   time_emulator = time_emulator,
                                                                                   train_batches = dataloader_opt, 
                                                                                   test_batches = dataloader_eval, 
                                                                                   device = device, 
                                                                                   learning_rate = config['learning_rate'], 
                                                                                   epochs = 5, # Shorter for hyperparameter validation
                                                                                   to_save = False) # Do not save models
        # Get list of time emulator losses
        te_loss = [x.item() for x in te_eval_losses]
        total_eval_loss += te_loss[-1] # Add final time emulator evaluation loss to total over folds

    avg_eval_loss = total_eval_loss / num_folds

    return {'loss': avg_eval_loss}