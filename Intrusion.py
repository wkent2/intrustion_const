import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from size_dist import get_davg_3phases,size_dist_basic
from numba import njit
import time
from pqdm.processes import pqdm
import argparse
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd

def parseargs():
    p = argparse.ArgumentParser(description="Calculates the constrictivity factor using the intrusion methodology.")
    p.add_argument('subvol_dir',help="Path to folder containing subvolume files (.npy)")
    p.add_argument('-c','-computation',type=str,choices=['serial','parallel'],default='parallel',help="Computational mode to use. Serial analyzes one subvolume at a time. Parallel uses all available processors to analyze multiple subvolumes at a time.")
    p.add_argument('-v','-verbose',type=bool,default=False,help="Whether or not to run with verbose updates.")
    p.add_argument('-f','-fast',type=bool,default=True,help="Whether or not to run in FAST mode (highly reccommended).")
    p.add_argument('-p','-plot',type=bool,default=False,help="Whether or not to plot PSDs. This is not fully implelemented.")
    p.add_argument('-nf','-num_faces',type=int,choices=[1,3,6],default=6,help="How many faces to intrude for each constricitivity caluclation.")

    args = p.parse_args()
    
    return args

def find_median_reversed(x, cdf):
    # Ensure CDF is monotonically decreasing
    if not np.all(np.diff(cdf) <= 0):
        raise ValueError("CDF values must be monotonically decreasing.")
    
    # Find the first index where CDF <= 0.5
    idx = np.searchsorted(cdf[::-1], 50, side="left")
    idx = len(cdf) - idx - 1  # Convert index for reversed array

    if idx == len(cdf) - 1:
        return x[-1]  # If 0.5 is above all values, return the last x
    elif idx == -1:
        return x[0]  # If 0.5 is below the first value, return the first x
    else:
        # Linear interpolation
        x0, x1 = x[idx], x[idx + 1]
        cdf0, cdf1 = cdf[idx], cdf[idx + 1]
        return x0 + (50 - cdf0) * (x1 - x0) / (cdf1 - cdf0)

def create_spherical_mask(R):
    """Creates a 3D binary mask for a sphere of radius R."""
    size = 2 * R + 1  # Ensure enough space
    center = R
    struct = np.zeros((size, size, size), dtype=bool)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center)**2 + (y - center)**2 + (z - center)**2 <= R**2:
                    struct[x, y, z] = True
    return struct

def get_seed_points(binary_array, face, R):
    """Finds seed points for a filling operation while ensuring proper padding and filtering."""
    
    # Define face indices and apply padding
    if face == 'z_min':
        binary_array = np.pad(binary_array, ((R, 0), (0, 0), (0, 0)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[R, :, :] == 1)  # Ensure only 1s are included
        seed_coords = np.column_stack((np.full(seed_points.shape[0], R), seed_points))  

    elif face == 'z_max':
        binary_array = np.pad(binary_array, ((0, R), (0, 0), (0, 0)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[-(R+1), :, :] == 1)
        seed_coords = np.column_stack((np.full(seed_points.shape[0], binary_array.shape[0] - (R+1)), seed_points))

    elif face == 'y_min':
        binary_array = np.pad(binary_array, ((0, 0), (R, 0), (0, 0)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[:, R, :] == 1)
        seed_coords = np.column_stack((seed_points[:, 0], np.full(seed_points.shape[0], R), seed_points[:, 1]))

    elif face == 'y_max':
        binary_array = np.pad(binary_array, ((0, 0), (0, R), (0, 0)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[:, -(R+1), :] == 1)
        seed_coords = np.column_stack((seed_points[:, 0], np.full(seed_points.shape[0], binary_array.shape[1] - (R+1)), seed_points[:, 1]))

    elif face == 'x_min':
        binary_array = np.pad(binary_array, ((0, 0), (0, 0), (R, 0)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[:, :, R] == 1)
        seed_coords = np.column_stack((seed_points[:, 0], seed_points[:, 1], np.full(seed_points.shape[0], R)))

    elif face == 'x_max':
        binary_array = np.pad(binary_array, ((0, 0), (0, 0), (0, R)), mode='constant', constant_values=1)
        seed_points = np.argwhere(binary_array[:, :, -(R+1)] == 1)
        seed_coords = np.column_stack((seed_points[:, 0], seed_points[:, 1], np.full(seed_points.shape[0], binary_array.shape[2] - (R+1))))

    else:
        raise ValueError("Invalid face. Choose from 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', or 'z_max'.")

    return binary_array, seed_coords 

def remove_padding(binary_array, face, R):
    """
    Removes padding applied to the binary_array in the get_seed_points function.
    
    Args:
    - binary_array (np.ndarray): The 3D binary array with padding.
    - face (str): The face where the padding was applied ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max').
    - R (int): The radius of the sphere used for padding.
    
    Returns:
    - np.ndarray: The binary array with padding removed.
    """
    if face == 'z_min':
        return binary_array[R:, :, :]  # Remove padding from the z_min face

    elif face == 'z_max':
        return binary_array[:-R, :, :]  # Remove padding from the z_max face

    elif face == 'y_min':
        return binary_array[:, R:, :]  # Remove padding from the y_min face

    elif face == 'y_max':
        return binary_array[:, :-R, :]  # Remove padding from the y_max face

    elif face == 'x_min':
        return binary_array[:, :, R:]  # Remove padding from the x_min face

    elif face == 'x_max':
        return binary_array[:, :, :-R]  # Remove padding from the x_max face

    else:
        raise ValueError("Invalid face. Choose from 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', or 'z_max'.")
        
def shorten(array,face):
    if face == 'z_min':
        return array[1:, :, :]  # Remove padding from the z_min face

    elif face == 'z_max':
        return array[:-1, :, :]  # Remove padding from the z_max face

    elif face == 'y_min':
        return array[:, 1:, :]  # Remove padding from the y_min face

    elif face == 'y_max':
        return array[:, :-1, :]  # Remove padding from the y_max face

    elif face == 'x_min':
        return array[:, :, 1:]  # Remove padding from the x_min face

    elif face == 'x_max':
        return array[:, :, :-1]  # Remove padding from the x_max face

    else:
        raise ValueError("Invalid face. Choose from 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', or 'z_max'.")

def intrusion(binary_array,centers,intruded,diam,face='z_min',verbose=False):
        
    # Calculate R (padding) for intrusion
    R = int(diam/2)
    
    # Creates sphere kernel for opening (element)
    sphere_kernel = create_spherical_mask(R)
    
    # Get seed coordinates
    seed_coords = np.argwhere(centers == 2)
    
    # Set previously rejected coordinates to 0
    centers[centers==2] = 0
    
    # Convert into list for queue operation
    queue = list(seed_coords)
    
    if verbose:
        print("Filling from face",face, "with",len(seed_coords),"front voxels and radius",R)
    
    #Test points until queue is empty
    while queue:
        
        # Pop queue
        test_center = queue.pop(0)
       
        # Checks that center hasn't already been checked
        if centers[tuple(test_center)] == 0:
            # Check if sphere can fit
            if check_contained(binary_array,sphere_kernel,test_center):
                # Label test_center as accepted (1) in centers array
                centers[tuple(test_center)] = 1 
                
                # Extract sphere center
                Z, Y, X = test_center
    
                # Define bounds for slicing
                z_min, z_max = Z - R, Z + R + 1
                y_min, y_max = Y - R, Y + R + 1
                x_min, x_max = X - R, X + R + 1
                
                # Open sphere
                # intruded[z_min:z_max, y_min:y_max, x_min:x_max][sphere_kernel] = 1
                apply_sphere_kernel(intruded, sphere_kernel, z_min, y_min, x_min)
                
                # Add voxels that are adjacent to the sphere center to the queue
                for neighbor in get_adjacent_voxels(centers.shape, test_center):
                    # Checks if tested and within phase
                    if centers[neighbor] == 0 and binary_array[neighbor]==1:
                        queue.append(neighbor)
                        
               
            else:
                # Label test_center as failed (2) in centers array
                centers[tuple(test_center)] = 2
        
     
    return intruded,centers

@njit
def apply_sphere_kernel(intruded, sphere_kernel, z_min, y_min, x_min):
    """Efficiently set values in 'intruded' where 'sphere_kernel' is True."""
    sz, sy, sx = sphere_kernel.shape  # Get sphere_kernel dimensions

    for dz in range(sz):
        for dy in range(sy):
            for dx in range(sx):
                if sphere_kernel[dz, dy, dx]:  # Only modify where sphere_kernel is True
                    intruded[z_min + dz, y_min + dy, x_min + dx] = 1


@njit
def get_adjacent_voxels(shape, coord):
    """Returns the 6-connected neighbors within array bounds using Numba for speed."""
    z, y, x = coord
    neighbors = []
    directions = np.array([(-1, 0, 0), (1, 0, 0),  # ±z
                           (0, -1, 0), (0, 1, 0),  # ±y
                           (0, 0, -1), (0, 0, 1)], dtype=np.int8)  # ±x

    for i in range(6):  # Loop over pre-defined directions
        dz, dy, dx = directions[i]
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
            neighbors.append((nz, ny, nx))

    return neighbors

@njit
def check_contained(mask, sphere_kernel, center):
    Z, Y, X = center
    radius = sphere_kernel.shape[0] // 2  # Assuming a cubic kernel with odd size

    # Define bounds
    z_min, z_max = Z - radius, Z + radius + 1
    y_min, y_max = Y - radius, Y + radius + 1
    x_min, x_max = X - radius, X + radius + 1

    # Ensure the sphere stays within bounds
    if (z_min < 0 or y_min < 0 or x_min < 0 or 
        z_max > mask.shape[0] or 
        y_max > mask.shape[1] or 
        x_max > mask.shape[2]):
        return False  # Out of bounds

    # Check if the sphere is fully contained in the pore space
    for dz in range(sphere_kernel.shape[0]):
        for dy in range(sphere_kernel.shape[1]):
            for dx in range(sphere_kernel.shape[2]):
                if sphere_kernel[dz, dy, dx] and not mask[z_min + dz, y_min + dy, x_min + dx]:
                    return False  # Found an excluded voxel

    return True  # Fully contained



def calc_iPSD(mask,diams,fast=True,verbose=False,num_faces=6):
    if verbose:
        print("Will compute with",num_faces,"faces")
    
    # Initialize array with all 6 faces
    if num_faces == 6:
        faces = ['z_min','z_max','y_min','y_max','x_min','x_max']
    elif num_faces == 3:
        faces = ['z_min','y_min','x_min']
    elif num_faces == 1:
        faces = ['z_min']
    
    # Initialize array for 
    voxels = np.zeros(shape=(len(faces),len(diams)))    
    voxels_final = np.zeros(len(diams))
    
    if fast:
        volume = np.count_nonzero(mask)
    
    for j,face in enumerate(faces):
        if verbose:
            print("Doing intrusion for",face,"face")
           
        # Get seed coordinates and padded array
        binary_array, seed_coords = get_seed_points(mask, face, int(np.amax(diams)/2))
            
        # Array to store intruded voxels and tested centers
        intruded = np.zeros(shape=binary_array.shape,dtype=np.uint8) # Used to calculate intruded volume
        centers = np.zeros(shape=binary_array.shape,dtype=np.uint8) # Stores tested voxels
            
        # Marks seeds in centers array
        for seed in seed_coords:
            centers[tuple(seed)] = 2
        
        for i,diam in enumerate(diams):
            
            # Intrude volume
            intruded,centers = intrusion(binary_array, centers,intruded,diam, face=face,verbose=verbose)
            
            # Remove padding from intrusion
            intruded_final = remove_padding(intruded, face, int(diam/2))
            
            # Counts intruded voxels 
            voxels[j,i] = np.count_nonzero(intruded_final)    
            if fast:
                if voxels[j,i] > 0.5*volume:
                    voxels[j,i+1:] = volume
                    if verbose:
                        print("Intruded 50% of the volume with R of",int(diam/2),"in",face)
                        print("Moving on to next face!")
                    break
            
            # Shorten all faces by 1 voxel to avoid 
            # marking padded area (saves time, I think)
            intruded = shorten(intruded,face)
            centers = shorten(centers,face)
            binary_array = shorten(binary_array,face)
    
    # Average all the faces together
    for i in range(len(voxels_final)):
        voxels_final[i] = np.mean(voxels[:,i])
    return voxels_final

def get_constrictivity(mask,fast=True,ret_dist=False,verbose=False,num_faces=6):
    
    if verbose:
        print("Computing diameter map for cPSD")
    
    # Compute sphere opening (diameter map)
    _,_,dmap = size_dist_basic(mask,padding=False,verbose=False)
    
    # Figure out maximum diameters and initialize array
    diams = np.arange(np.amax(dmap),0,-2)
    
    if verbose:
        print("Computing cPSD")
        
    # Compute the continuous particle size distribution (cPSD)
    cPSD = np.array([np.count_nonzero(dmap[dmap>=d]) for d in diams]) 
    
    if verbose:
        print("Performing intrusions for iPSD")
    
    # Compute the intrusion particle size distribution (iPSD)
    # diams[1:] skips diameter=1, not necessary 
    iPSD = calc_iPSD(mask,diams,fast=fast,verbose=verbose,num_faces=num_faces)
    
    # Reverse order
    diams = diams[::-1]
    cPSD = cPSD[::-1]
    iPSD = iPSD[::-1]

    
    # Normalize
    cPSD = 100*cPSD/max(cPSD)
    iPSD = 100*iPSD/max(iPSD)
    
    if verbose:
        print("Finding median values and computing constrictivity")
    
    # Get constriction and average radii
    rmin = find_median_reversed(diams,iPSD)
    rmax = find_median_reversed(diams,cPSD)
    
    # Compute constrictivity factor
    const = (rmin/rmax)**2
    
    if ret_dist:
        return const, cPSD, iPSD, diams,rmin,rmax
    else:
        return const

def subvol_const(vol,fast=True,plot_dists=False,verbose=False,num_faces=6):
    
    if verbose:
        if fast:
            print("Running in FAST mode")
            
        print("Finding unique phases")
        
    
    # Get unique phases in vol
    phases = np.unique(vol)
    
    if verbose:
        print("Found",len(phases),"phases")
    
    # Initialize results array for constrictivity values 
    const = np.zeros(len(phases))
    
    if verbose:
        print("Beginning constrictivity calculations...")
        begin = time.time()
    
    # Calculate the constrictivity for each phase
    for i,phase in enumerate(phases):
        
        if verbose:
            print("Computing phase",phase,"constrictivity...")
        
        # Get phase mask
        mask = np.zeros_like(vol)
        mask[vol==phase] = 1
        
        # Compute constrictivity factor
        if plot_dists:
            const[i],cPSD, iPSD, diams,rmin,rmax = get_constrictivity(mask,fast=fast,ret_dist=True,verbose=verbose,num_faces=num_faces)
            
            plt.close('all')
            plt.figure(figsize=(7,5))
            plt.subplot(1,1,1)
            plt.plot(diams,cPSD,'k--',label='c-PSD')
            plt.plot(diams,iPSD,'k-',label='i-PSD')
            plt.axvline(rmin,linewidth=0.5,color='black')
            plt.axvline(rmax,linewidth=0.5,color='black')
            plt.xlabel('Pore Diameter [Voxels]',fontsize=14)
            plt.ylabel('Cumulative Frequency [%]',fontsize=14)
            title_text = 'Phase '+str(phase)+' PSDs'
            plt.title(title_text,fontsize=14)
            savename = './'+str(i)+'_PSDs.png'
            plt.savefig(savename,dpi=300)
        else:
            const[i] = get_constrictivity(mask,fast=fast,ret_dist=False,verbose=verbose,num_faces=num_faces)
        
    if verbose:
        print("Done!")
        end = time.time()
        print("It took",round((end-begin)/60,2),"minutes for one subvolume")
    
    return const,phases
        
def subvol_wrapper(arguments):
    
    # Extract arguments
    filepath,fast,plot_dists,verbose,num_faces = arguments
    
    try:
        # Load volume
        vol = np.load(filepath)

        try:
            # Compute constrictivities
            consts,phases = subvol_const(vol,fast=fast,plot_dists=plot_dists,verbose=verbose,num_faces=num_faces)
        except:
            # Return nan values if something went wrong
            consts,phases = np.array([np.nan,np.nan,np.nan]),np.array([0,1,2])
            
    except:
        consts,phases = np.array([np.nan,np.nan,np.nan]),np.array([0,1,2]) 

    result = np.array([consts,phases])

    return result
    
def process_results(results):

    # Get unique number of phases
    unique_phases = []

    # Iterate through all intrusion results
    for res in results:
        
        try:
            # Pull out results
            consts,phases = res 

            # Iterate through all phases
            for i in range(len(phases)):
                # Check if new phase present
                if phases[i] not in unique_phases:
                    # Add new phase
                    unique_phases.append(phases[i])
        except:
            pass

    # Sort phases
    unique_phases = np.array(sorted(unique_phases))

    value_to_index = {value: idx for idx, value in enumerate(unique_phases)}

    # Initialize array to store final results
    final_consts = np.zeros(shape=(len(results),len(unique_phases)))

    # Iterate through all intrusion results
    for i,res in enumerate(results):
        
        try:
            # Pull out results
            consts,phases = res 


            # Iterates thr-ough all phases
            for j in range(len(phases)):
                # Assigns beta value to the correct position
                final_consts[i,value_to_index[phases[j]]] = consts[j]
        except:
            pass

    # Give all 0 values nan
    final_consts[final_consts==0]=np.nan

    # Creates column labels
    column_names = ['C'+str(int(phase)) for phase in unique_phases]

    return final_consts, column_names

if __name__ == "__main__":
    
    # Parse command line arguments
    args = parseargs()
    
    # Get all files
    files = [file for file in os.listdir(args.subvol_dir) if file.endswith('.npy')]
    
    # Create arguments
    filepaths = [os.path.join(args.subvol_dir,file) for file in os.listdir(args.subvol_dir) if file.endswith('.npy')]
    arguments = [(filepath,args.f,args.p,args.v,args.nf) for filepath in filepaths]
    
    print("Running in",args.c,"mode")
    print("Will intrude",args.nf,"faces per subvolume")
    # Run in serial mode
    if args.c == 'serial':
        constrictivities = []
        for arg in tqdm(arguments):
            constrictivities.append(subvol_wrapper(arg))
        
        
    
    # Run in parallel mode
    elif args.c == 'parallel':
        constrictivities = pqdm(arguments,subvol_wrapper,n_jobs=mp.cpu_count())
    
    # Handle errors and number of phases 
    final_consts, column_names = process_results(constrictivities)
    
    # print(final_consts.shape)
    # print(final_consts)
    # Compile data into DataFrame and save as .csv file
    data = pd.DataFrame(final_consts,columns=column_names,index=files)
    
    savepath = os.path.join(args.subvol_dir,'constrictivity_results.csv')
    
    header1 = str(args.nf)+" faces were used for constrictivity caluclation."
    if args.f:
        fstmd = "FAST"
    else:
        fstmd = "SLOW"
    header2 = "Code was run in " + fstmd + " mode"
    extra_headers = [
    header1,
    header2,
    ]
    
    # Save with extra headers
    with open(savepath, "w") as f:
        for line in extra_headers:
            f.write(line + "\n")  # Write each extra header line
        data.to_csv(f, index=True,index_label="Filename")  # Append DataFrame to the file
    
    print("Done! Saved results to",savepath)

