# Module containing functions to help sample space of physical parameters and initial conditions
# and generate files to run Chempl to evolve the resulting chemical networks

# Import statements
from scipy.stats import qmc # Multi-dimensional quasi-Monte Carlo sampling from Scipy
import numpy as np # Numpy for math
import pandas as pd # Pandas for dataframe handling
from pathlib import Path # To make nexted directories to store chempl files

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
    T_0 = 12.2 * (G0_UV ** 0.25) # In K
    opt_depth = 1e-3 # Optical depth at 100 micrometers
    nu_0 = 3e15 # Characteristic frequency
    T_dust = np.power(8.9e-11 * G0_UV * nu_0 * np.exp(-1.8 * A_V) + 
                      (2.78 ** 5) + 3.4e-2 * (0.42 - opt_depth * (T_0 ** 6) * 
                                              np.log(3.5e-2 * opt_depth * T_0)),
                      0.2)
    # Calculate f_H2 from the fitting formula of Polzin et al. 2024
    # Needed parameters
    R0 = 3.5e-17 # H2 formation rate on dust grains (Wolfire et al. 2008)
    dust_to_gas_ratio = 1e-2 # dust-to-gas mass ratio (chempl default value)
    # Scale 1000A value of U_MW = J_1000A/J_MW from Draine 1978 by appropriate G0 from Draine ISM book
    U_MW = 0.662 * (G0_UV / 1.69) 
    # Equation 4
    Myr_in_s = 3.16e13 # seconds per Myr
    Q = 6 * R0 * n_gas * (metallicity / 0.2) ** (1.3)  * Myr_in_s
    # Equation 5
    f_m = 1 - np.exp(-1 * Q)
    # Equation 2
    f_H2_max = 1 / (1 + 2 * (1 - f_m) / f_m)
    # Equation 7
    a = (34.7 * (U_MW ** 0.32)) - 2.25 * ((dust_to_gas_ratio / 0.0199) ** 0.3)
    b = -53.9 * (U_MW ** 0.31)
    c = dust_to_gas_ratio / (0.2 * 0.0199) # Equation 8
    n_tr = b - (a * np.log10(dust_to_gas_ratio)) + c # Transition density
    # Equations 5-6
    x = 7.6 * (metallicity ** 0.25) * np.log(n_gas / n_tr)
    # Equation 1
    f_H2 = f_H2_max / (1 + np.exp(7.42 - x + np.log(f_H2_max)))
    # Calculate H2 column density
    N_H2 = N_H * f_H2
    # Write the physical parameters to the file
    f = open(dir_path + 'phy_params.dat', 'w') # Open file in writing mode
    f.write('T_gas = ' + str(T_gas) + '\n') # Write given gas temperature
    f.write('n_gas = ' + str(n_gas) + '\n') # Write given gas density
    f.write('G0_UV = ' + str(G0_UV) + '\n') # Write given radiation field strength
    f.write('dust2gas_mass = ' + str(dust_to_gas_ratio) + '\n') # Write default chempl parameters we don't change
    f.write('chi_cosmicray = 1.0\n') 
    f.write('chi_Xray = 0.0\n') 
    f.write('dust_material_density = 2.0\n') 
    f.write('dust_site_density = 1.0e15\n') 
    f.write('dust_radius = 0.1e-4\n') 
    f.write('dust_albedo = 0.6\n') 
    f.write('mean_mol_weight = 1.4\n') 
    f.write('chemdesorption_factor = 0.05\n') 
    f.write('v_km_s = 14.5\n') 
    f.write('dv_km_s = 1.0\n') 
    f.write('Ncol_H2 = ' + str(N_H2) + '\n') # Write calculated H2 column density
    f.write('Av = ' + str(A_V) + '\n') # Write calculated dust extinction
    f.write('T_dust = ' + str(T_dust) + '\n') # Write calculated dust temperature
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
                          'F' : metallicity * 10 ** (4.56 - 12), # Flourine abundance
                          'Na' : metallicity * 10 ** (6.24 - 12), # Sodium abundance
                          'Mg' : metallicity * 10 ** (7.60 - 12), # Magnesium abundance
                          'Si' : metallicity * 10 ** (7.51 - 12), # Silicon abundance
                          'P' : metallicity * 10 ** (5.41 - 12), # Phosphorus abundance
                          'S' : metallicity * 10 ** (7.12 - 12), # Sulfur abundance
                          'Cl' : metallicity * 10 ** (5.50 - 12), # Chlorine abundance
                          'Fe' : metallicity * 10 ** (7.50 - 12) # Iron abundance
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
def sampling(num_samples, samples_dir_path, batch_num = 0, 
             T_gas_min = 1e1, T_gas_max = 1e4, n_gas_min = 1e1, 
             n_gas_max = 1e6, G0_UV_min = 1e-2, G0_UV_max = 1e3):
    '''
    Inputs:
    num_samples (int): Number of samples per batch
    samples_dir_path (str): Directory where sample files will be written
    batch_num (int): Number of batches sampled previously (to ensure reproducibility, defaults to 0)
    T_gas_min (float): Lower limit of T_gas range to sample, in K (defaults to 10)
    T_gas_max (float): Upper limit of T_gas range to sample, in K (defaults to 10^4)
    n_gas_min (float): Lower limit of n_gas range to sample, in cm^-3 (deafults to 10)
    n_gas_max (float): Upper limit of n_gas range to sample, in cm^-3 (defaults to 10^6)
    G0_UV_min (float): Lower limit of G0_UV range to sample (defaults to 10^-2)
    G0_UV_max (float): Upper limit of G0_UV range to sample (defaults to 10^6)
    Outputs:
    scaled_samples (d*num_samples ndarray): multidimensional array with num_samples instances of d samples
    '''

    # Initialize the sampler (specify random seed for reproducibility)
    sampler = qmc.LatinHypercube(d = 4, seed = 3395)
    # Fastforward by num_samples * batch_num to ensure consistency
    sampler.fast_forward(n = num_samples * batch_num)
    # Generate num_samples samples from the 4-dimensional unit hypercube
    unscaled_samples = sampler.random(n = num_samples)

    # Scale the samples to the approprate intervals 
    scaled_samples = unscaled_samples # Initialize an object to hold the scaled samples
    # Iterate through samples (since cell_thickness_pc range is defined based on T_gas, n_gas values for each sample)
    for index in range(num_samples):
        unscaled_sample = unscaled_samples[index]
        # Linearly rescale x in [0, 1] to get log(sample) in correct range
        log_T_gas = np.log10(T_gas_min) + unscaled_sample[0] * (np.log10(T_gas_max) - np.log10(T_gas_min))
        log_n_gas = np.log10(n_gas_min) + unscaled_sample[1] * (np.log10(n_gas_max) - np.log10(n_gas_min))
        log_G0_UV = np.log10(G0_UV_min) + unscaled_sample[2] * (np.log10(G0_UV_max) - np.log10(G0_UV_min))
        # Calculate Jeans length in pc
        jeans_length_pc = 17 * np.sqrt((10 ** (log_T_gas)) / (10 ** (log_n_gas)))
        # Get min, max values to sample in [0.01, 10] * jeans_length_pc
        log_ct_min = np.log10(jeans_length_pc / 100)
        log_ct_max = np.log10(jeans_length_pc * 10)
        # Linearly rescale x in [0, 1] to get log(cell thickness) in correct range
        log_ct_pc = log_ct_min + unscaled_sample[3] * (log_ct_max - log_ct_min)

        # Exponentiate the values and store them in scaled_samples
        scaled_samples[index] = [10 ** log_T_gas, 10 ** log_n_gas, 10 ** log_G0_UV, 10 ** log_ct_pc]
    
        # Create the chempl files corresponding to this file
        # Create a directory corresponding to this file index (starting at batch_num * num_samples)
        path_name = samples_dir_path + str(index + batch_num * num_samples).zfill(6) + '/'
        directory_path = Path(path_name)
        directory_path.mkdir(exist_ok = True)
        # Write a .dat file containing the sampled parameters (needed as inputs to the emulator)
        sample_df = pd.Series({'T_gas' : 10 ** log_T_gas, 'n_gas' : 10 ** log_n_gas,
                               'G0_UV' : 10 ** log_G0_UV, 'cell_thickness_pc' : 10 ** log_ct_pc})
        sample_df.to_string(buf = path_name + 'initial_abundances.dat', header = False)
        # Generate the needed chempl files
        generate_chempl_files(10 ** log_T_gas, 10 ** log_n_gas, 10 ** log_G0_UV, 10 ** log_ct_pc, # Sampled parameters
                              metallicity = 1, dir_path = path_name)

    # Return the scaled samples
    return scaled_samples