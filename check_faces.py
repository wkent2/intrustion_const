import numpy as np
import os

import argparse
from tqdm import tqdm
from Intrusion import get_seed_points
from pqdm.processes import pqdm
import multiprocessing as mp
import pandas as pd


def parseargs():
    p = argparse.ArgumentParser(description="Checks if there are any subvolumes with faces not containing all subvolume phases.")
    p.add_argument('subvol_dir',help="Path to folder containing subvolume files (.npy)")
    p.add_argument('-c','-computation',type=str,choices=['serial','parallel'],default='parallel',help="Computational mode to use. Serial analyzes one subvolume at a time. Parallel uses all available processors to analyze multiple subvolumes at a time.")

    args = p.parse_args()
    
    return args



def check_subvol(filepath):

	
	vol = np.load(filepath)

	phases = np.unique(vol)

	faces = ['z_min','z_max','y_min','y_max','x_min','x_max']

	contains_phase = np.zeros(shape=len(phases))

	# Check each phase in the subvolume
	for i,phase in enumerate(phases):

		# Creates a mask of the phase
		mask = np.zeros_like(vol)
		mask[vol==phase] = 1

		present = 1
		# Checks each face
		for j,face in enumerate(faces):
           
			# Get seed coordinates and padded array
			binary_array, seed_coords = get_seed_points(mask, face, 1)
			if len(seed_coord)==0:
				present = 0
				break

		contains_phase[i] = present

	print(contains_phase)

	return contains_phase



if __name__ == "__main__":
    
    # Parse command line arguments
    args = parseargs()
    
    # Get all files
    files = [file for file in os.listdir(args.subvol_dir) if file.endswith('.npy')]
    
    # Create arguments
    filepaths = [os.path.join(args.subvol_dir,file) for file in os.listdir(args.subvol_dir) if file.endswith('.npy')]

    # Run in serial mode
    if args.c == 'serial':
        contained = []
        for filepath in tqdm(filepaths):
            contained.append(check_subvol(filepath))
    
    # Run in parallel mode
    elif args.c == 'parallel':
        contained = pqdm(filepaths,check_subvol,n_jobs=mp.cpu_count())


    # Compiles results
    contained = np.concatenate(contained,axis=0)

    # Formats results into dataframe
    data = pd.DataFrame(contained,columns=['P1','P2','P3'],index=files)

    # Creates savepath
    savepath = os.path.join(args.subvol_dir,'phase_checks.csv')

    # Saves data to .csv file
    data.to_csv(savepath)
    
    print("Done! Saved results to",savepath)

