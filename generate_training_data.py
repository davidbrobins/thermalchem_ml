# Script to generate training data
# Usage: python generate_training_data.py samples_dir/ num_samples prev_samples > gtd_log.txt

# Import statements
import sys # Command line arguments
import time # Timing
from multiprocessing import Pool # Pool for running chempl in parallel
import subprocess # subprocess to run bash commands
import sampling_tools # Sampling functions in sampling_tools.py
import numpy as np # Numpy for math

# Unpack command line arguments
# Name of this file, directory where samples will be stored, number of samples, number of samples generated previously
(pyfilename, samples_dir, num_samples, prev_samples) = sys.argv 

# Run the sampling
samples = sampling_tools.sampling(num_samples, samples_dir, prev_samples)
# Print to check
print(samples)

# Define a function to run chempl on a given sample index
def run_chempl(sample_index):
    '''
    Input: 
    sample_index (int) : Index of the sample 
    Outputs:
    None
    '''

    # Change working location to chempl directory (modify as needed for your working setup)
    subprocess.run(['cd', '../chempl'])
    # Get location of run file containing locations of needed parameter files for chempl
    run_file = samples_dir + str(sample_index).zfill(6) + '/run_file.dat'
    print('Run file being used: ', run_file)
    # Run chempl on run_file
    subprocess.run(['./re', run_file], capture_output = True)

# Run chempl in parallel across 8 processors
if __name__ == "__main__": 
    # Set up the parallel processing
    pool = Pool(processes = 8) # Modify as needed for your system

    # File indices run from prev_samples to prev_samples + num_samples - 1
    sample_indices = np.arange(prev_samples, prev_samples + num_samples)

    # Apply run_chempl (defined above) on index using the parallel processing pool
    results = pool.map(run_chempl, sample_indices)

    # End the pool and recombine each process
    pool.close()
    pool.join()