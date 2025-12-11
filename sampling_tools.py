# Module containing functions to help sample space of physical parameters and initial conditions
# and generate files to run Chempl to evolve the resulting chemical networks

# Import statements
from scipy.stats import qmc # Multi-dimensional quasi-Monte Carlo sampling from Scipy
import numpy as np # Numpy for math
import pandas as pd # Pandas for dataframe handling

# Function to generate files for Chempl run (physical parameters, initial conditions, and a runfile) from sampled parameters
def generate_chempl_files(T_gas, n_gas, G0_UV, cell_thickness_pc, metallicity, dir_path, specific_abundances = {}):
    '''
    Inputs:
    T_gas (float): Gas temperature [K]
    n_gas (float): Gas (hydrogen) number density [cm^-3]
    G0_UV (float): Radiation field strength (in units of Draine field)
    cell_thickness_pc (float): Cell thickness [pc]
    metallicity (float): Initial metallicity in units of Z/Z_sun
    dir_path (str): Path to the directory where the files will be written (should be an absolute path)
    specific_abundances (dir): Dictionary containing any desired initial chemical abundances that differ from the overall metallicity (defaults to an empty dictionary)
    Outputs:
    None (writes file containing physical parameters for chempl run to file_path)
    '''

    # Calculate derived parameters
    # Convert cell thickness to cm
    pc_to_cm = 3.1e18
    cell_thickness_cm = cell_thickness_pc * pc_to_cm
    # Calculate hydrogen column density in cm^-2
    N_H = n_gas * cell_thickness_cm
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
    # Calculate f_H2 from the fitting formula of Polzin et al. 2024
    # Needed parameters
    metallicity = 1 # Z/Z_sun
    R0 = 3.5e-17 # H2 formation rate on dust grains (Wolfire et al. 2008)
    dust_to_gas_ratio = 1e-2 # dust-to-gas mass ratio (chempl default value)
    # Scale 1000A value of U_MW = J_1000A/J_MW from Draine 1978 by appropriate G0 from Draine ISM book
    U_MW = 0.662 * (G0_UV / 1.69) 
    # Equation 4
    Q = 6 * R0 * n_gas * (metallicity / 0.2) ** (1.3) # In Myr
    # Equation 5
    f_m = 1 - np.exp(-1 * Q)
    # Equation 2
    f_H2_max = (1 + 2 * (1 - f_m) / f_m) ** (-1)
    # Equation 7
    a = (34.7 * (U_MW ** 0.32)) - 2.25 * ((dust_to_gas_ratio / 0.0199) ** 0.3)
    b = -53.8 * (U_MW ** 0.31)
    c = dust_to_gas_ratio / (0.2 * 0.0199)
    n_tr = b - (a * np.log10(dust_to_gas_ratio)) + c # Transition density
    # Equations 5-6
    x = 7.6 * (metallicity ** 0.25) * np.log(n_gas / n_tr)
    # Equation 1
    f_H2 = f_H2_max / (1 + np.exp(7.42 - x + np.log(f_H2_max)))
    # Calculate H2 column density
    N_H2 = N_H * f_H2
    # Write the physical parameters to the file
    f = open(dir_path + 'phy_params.dat', 'w') # Open file in writing mode
    f.write('T_gas = ' + str(T_gas) + '\n') # Write gas temperature
    f.write('n_gas = ' + str(n_gas) + '\n') # Write gas density
    f.write('G0_UV = ' + str(G0_UV) + '\n') # Write radiation field strength
    f.write('Ncol_H2 = ' + str(N_H2) + '\n') # Write H2 column density
    f.write('Av = ' + str(A_V) + '\n') # Write dust extinction
    f.write('T_dust = ' + str(T_dust) + '\n') # Write dust temperature
    f.write('t_max_year = 1e8') # Write t_stop = 10^8 yr
    f.close() # Close the file

    # Set the initial abundances
    # by scaling the solar metal abundances from Asplund et al. 2009 by the overall metallicity
    initial_abundances = {'H2' : f_H2, # H2 abundance is just the H2 fraction
                          'H' : 1 - f_H2, # All remainoing hydrogen is in atomic form
                          'He' : 10**(10.93 - 12), # Fractional helium abundance
                          'C' : metallicity * 10 ** (8.43 - 12), # Carbon abundance (all metals below scaled by the overall metallicity)
                          'N' : metallicity * 10 ** (7.83 - 12), # Nitrogen abundance
                          'O' : metallicity * 10 ** (8.69 - 12), # Oxygen abundance
                          'F' : metallicity * 10 ** (4.56 -12), # Flourine abundance
                          'Na' : metallicity * 10 ** (6.24 - 12), # Sodium abundance
                          'Mg' : metallicity * 10 ** (7.60 -12), # Magnesium abundance
                          'Si' : metallicity * 10 ** (7.51 - 12), # Silicon abundance
                          'P' : metallicity * 10 ** (5.41 - 12), # Phosphorus abundance
                          'S' : metallicity * 10 ** (7.12 - 12), # Sulfur abundance
                          'Cl' : metallicity * 10 ** (5.50 - 12), # Chlorine abundance
                          'Fe' : metallicity * 10 ** (7.50 -12) # Iron abundance
    }
    # Check for any additional or modified abundances in the optional dictionary
    for species in specific_abundances:
        # Set the abundance of that species as specified (overwrites or creates a new entry as needed)
        initial_abundances[species] = specific_abundances[species]
    # Convert the initial abundances to a pandas Series
    initial_abundances = pd.Series(initial_abundances)
    # Write this series to a file
    initial_abundances.to_string(buf = dir_path + 'initial_abundances.dat', header = False)

    # Write the run file
    f = open(dir_path + 'run_file.dat', 'w') # Create and open the file
    f.write('f_phy_params = ' + dir_path + 'phy_params.dat\n') # Specify the full path to the physical parameters file created above
    f.write('f_reactions = rate12_combined.dat\n') # Specify the reaction rates file
    f.write('f_initial_abundances = ' + dir_path + 'initial_abundances.dat\n') # Specify the path to the initial conditions file created above
    f.write('f_enthalpies = Species_enthalpy.dat\n') # Specify the enthalpies file
    f.write('f_record = ' + dir_path + 'results.dat') # Specify the path where the results file will be created
    f.close() # Close the file

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
