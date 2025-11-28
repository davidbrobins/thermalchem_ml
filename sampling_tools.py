# Module containing functions to help sample space of physical parameters and initial conditions
# and generate files to run Chempl to evolve the resulting chemical networks

# Import statements
from scipy.stats import qmc # Multi-dimensional quasi-Monte Carlo sampling from Scipy
import numpy as np

# Function to generate physical parameters file for Chempl from sampled parameters
def generate_phy_params_file(T_gas, n_gas, G0_UV, f_H2, cell_thickness_pc, file_path):
    '''
    Inputs:
    T_gas (float): Gas temperature [K]
    n_gas (float): Gas (hydrogen) number density [cm^-3]
    G0_UV (float): Radiation field strength (in units of Draine field)
    f_H2 (float): Fraction of hydrogen in the form of H2 (in the interval [0, 1])
    cell_thickness_pc (float): Cell thickness [pc]
    file_path (str): Path to the file where the physical parameters will be written (must include the file name)
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