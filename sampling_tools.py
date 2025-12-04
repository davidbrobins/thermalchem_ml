# Module containing functions to help sample space of physical parameters and initial conditions
# and generate files to run Chempl to evolve the resulting chemical networks

# Import statements
from scipy.stats import qmc # Multi-dimensional quasi-Monte Carlo sampling from Scipy
import numpy as np # Numpy for math
import pandas as pd # Pandas for dataframe handling

# Function to generate physical parameters file for Chempl from sampled parameters
def generate_phy_params_file(T_gas, n_gas, G0_UV, f_H2, cell_thickness_pc, file_path):
    '''
    Inputs:
    T_gas (float): Gas temperature [K]
    n_gas (float): Gas (hydrogen) number density [cm^-3]
    G0_UV (float): Radiation field strength (in units of Draine field)
    f_H2 (float): Initial fraction of hydrogen in the form of H2 (in the interval [0, 1])
    cell_thickness_pc (float): Cell thickness [pc]
    file_path (str): Path to the file where the physical parameters will be written (must include the file name)
    Outputs:
    None (writes file containing physical parameters for chempl run to file_path)
    '''

    # Calculate derived parameters
    # Convert cell thickness to cm
    pc_to_cm = 3.1e18
    cell_thickness_cm = cell_thickness_pc * pc_to_cm
    # Calculate hydrogen column density in cm^-2
    N_H = n_gas * cell_thickness_cm
    # Calculate H2 column density
    N_H2 = N_H * f_H2
    # Calculate dust extinction A_V
    N_H_to_A_V_ratio = 5.3e-22 # Draine "Physics of the Interstellar and Intergalactic Medium", equation 21.7
    A_V = N_H * N_H_to_A_V_ratio
    # Calculate dust temperature (Tielens "The Physics and Chemistry of the Interstellar Medium", equation 9.18)
    T_0 = 12.2 * (G0_UV ** 0.25)
    opt_depth = 1e-3
    nu_0 = 3e15
    T_dust = np.power(8.9e-11 * G0_UV * nu_0 * np.exp(-1.8 * A_V) + 
                      (2.78 ** 5) + 3.4e-2 * (0.42 - opt_depth * (T_0 ** 6) * 
                                              np.log(3.5e-2 * opt_depth * T_0)),
                      0.2)
    
    # Write the physical parameters to the file
    f = open(file_path, 'w') # Open file in writing mode
    f.write('T_gas = ' + str(T_gas) + '\n') # Write gas temperature
    f.write('n_gas = ' + str(n_gas) + '\n') # Write gas density
    f.write('G0_UV = ' + str(G0_UV) + '\n') # Write radiation field strength
    f.write('Ncol_H2 = ' + str(N_H2) + '\n') # Write H2 column density
    f.write('Av = ' + str(A_V) + '\n') # Write dust extinction
    f.write('T_dust = ' + str(T_dust) + '\n') # Write dust temperature
    f.close() # Close the file

# Function to generate ICs file for Chempl from the sampled parameters
def generate_ics_file(f_H2, x_N, x_C, x_O, x_S, x_Si, x_Na, x_Mg, x_Fe, x_P, x_F, x_Cl, file_path):
    '''
    Inputs:
    f_H2 (float): Initial fraction of hydrogen in (neutral) molecular form (in the interval [0, 1])
    x_N (float): Initial abundance of nitrogen (all abundances x_i = n_i/n_H)
    x_C (float): Initial abundance of carbon
    x_O (float): Initial abundance of oxygen
    x_S (float): Initial abundance of sulfur
    x_Si (float): Initial abundance of silicon
    x_Na (float): Initial abundance of sodium
    x_Mg (float): Initial abundance of magnesium
    x_Fe (float): Initial abundance of iron
    x_P (float): Initial abundance of phosphorus
    x_F (float): Initial abundance of flourine
    x_Cl (float): Initial abundance of chlorine
    file_path (str): Path to the file where the physical parameters will be written (must include the file name)
    Outputs:
    None (writes file containing physical parameters for chempl run to file_path)
    '''

    # Package all abundances as a pandas Series 
    abundances = pd.Series({'H2': f_H2, # Helium?
                            'N': x_N, 'C': x_C, # Do any of the ions need to be charged?
                            })
    # Write the abundances to a file
    abundances.to_string(buf = file_path, header = False)

# Function to run the sampler
def sampling(num_samples, batch_num = 0):
    '''
    Inputs:
    num_samples (int): Number of samples per batch
    batch_num (int): Number of batches sampled previously (to ensure reproducibility, defaults to 0)
    Outputs:
    scaled_samples (d*num_samples ndarray): multidimensional array with num_samples instances of d samples
    '''

    # Initialize the sampler (specify random seed for reproducibility)
    sampler = qmc.LatinHypercube(d = 10, seed = 3395)
    # Fastforward by num_samples * batch_num to ensure consistency
    sampler.fast_forward(n = num_samples * batch_num)
    # Generate num_samples samples from the d-dimensional unit hypercube
    unscaled_samples = sampler.random(n = num_samples)

    # Define lower and upper limits to rescale the d-dimensional unit hypercube to our parameter space
    # In order, the parameters sampled are:
    # [log(T_gas), log(n_gas), log(G0_UV), log(f_H2), log(cell_thickness_pc), 
    #  log(x_N), log(x_C), log(x_O), log(x_S), log(x_Si), log(x_Na), log(x_Mg), 
    #  log(x_Fe), log(x_P), log(x_F), log(x_Cl)]
    lower_bounds = []
    upper_bounds = []
    # Scale the samples
    scaled_samples = qmc.scale(unscaled_samples, lower_bounds, upper_bounds)

    # Return the scaled samples
    return scaled_samples

# Function to run the sampling procedure and generate all chempl files for each sample
def generate_chempl_files(num_samples, batch_num, file_directory):
    '''
    Inputs:
    num_samples (int): Number of samples per batch
    batch_num (int): Number of batches sampled so far
    file_directory (str): Filepath to store chempl files
    Outputs:
    None (chempl files corresponding to the sampled parameters will be written to file_directory)
    '''

    # Call the sampler to generate num_samples samples

    

    






