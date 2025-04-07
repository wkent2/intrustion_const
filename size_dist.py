#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:54:25 2024

@author: williamkent
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import time

def tstring_h(t):
	"""tstring_h (h stands for human) converts a time, in seconds, to either
		MM:SS, HH:MM, or DD:HH, depending on how many seconds it is. Useful when
		using "toc" to write an easy-to-read output."""

	if t < 99:
		return '%.3fs' % t
	elif t < 3600: #minutes. Do MM:SS
		m = '%im' % (t/60)
		s = '%.2fs' % (t%60)
		return m+s
	elif t < 3600*26: #HH:MM
		h = '%ih' % (t/3600)
		m = '%im' % round((t % 3600)/60)
		return h+m
	else: #days!
		d = '%id' % (t/3600/24)
		h = '%.2fh' % ((t % (3600*24))/3600)
		return d+h

def KernelCircle(R):
    '''
    Generates a circular kernel of radius R. 
    Output is a #D array with 0s everywhere, 1s to mark the sphere
    '''
    RR_ = list(range(-R,R+1))
    x, y,z = np.meshgrid(RR_,RR_,RR_)
    x, y,z= x.astype(float), y.astype(float), z.astype(float)
    
    circle = np.power(x/R,2) + np.power(y/R,2) + np.power(z/R,2) <= 1
    return circle

def KernelCircle2d(R):
    '''
    Generates a circular kernel of radius R. 
    Output is a 2D array with 0s everywhere, 1s to mark the sphere
    '''
    RR_ = list(range(-R,R+1))
    x, y = np.meshgrid(RR_,RR_)
    x, y= x.astype(float), y.astype(float)
    
    circle = np.power(x/R,2) + np.power(y/R,2) <= 1
    return circle

def erode(data, strel):
	"""
	data_eroded = erode(data, strel)

	Erodes DATA using structural element STREL using a frequency-space convolution
	This is much faster and more memory efficient than scipy.ndimage.binary_erosion
	for large STRELs, and gives identical results
	"""

	return fftconvolve(data,strel,'same') > (np.count_nonzero(strel)-0.5)

def dilate(data,strel):
	"""
	data_dilated = dilate(data, strel)
	Dilates DATA using structural element STREL using a frequency-space convolution
	This is much faster and more memory efficient than scipy.ndimage.binary_dilation
	for large STRELs and gives identical results.
	"""
	#return scipy.ndimage.morphology.binary_dilation(data,structure=strel)
	return fftconvolve(data,strel,'same')>0.5

def get_padding(data):

	# Find the maximum diameters on each face
    maxes = np.zeros(6)

    maxes[0] = np.amax(size_dist_basic2d(data[0,:,:],verbose=False)[2]) #-z
    maxes[1] = np.amax(size_dist_basic2d(data[-1,:,:],verbose=False)[2]) #+z
    maxes[2] = np.amax(size_dist_basic2d(data[:,0,:],verbose=False)[2]) #-y
    maxes[3] = np.amax(size_dist_basic2d(data[:,-1,:],verbose=False)[2]) #+y
    maxes[4] = np.amax(size_dist_basic2d(data[:,:,0],verbose=False)[2]) #-x
    maxes[5] = np.amax(size_dist_basic2d(data[:,:,-1],verbose=False)[2]) #+x

    maxR = int(np.amax(maxes)/2)

    return np.pad(data, pad_width=maxR, mode='constant', constant_values=1),maxR


def size_dist_basic(data,padding=False,verbose=True):
    '''
    Data is a 3d matrixs of 1s and 0s, where 1s are the phase of interest
    '''

    if padding:
    	print("Pre-padding",data.shape)
    	data,pad_val = get_padding(data)
    	print("Post-padding",data.shape)

    maxR = np.max(np.shape(data))
    
    # Initialize output
    diams_3d = np.zeros(np.shape(data),dtype=int)
    radii = []
    
    for r in range(1,maxR+1):
        t = time.time()
        sphere = KernelCircle(r)
        
        #Method: erode then dilate (the definition of opening) using custom fft-convolution method
        data_temp = erode(data,sphere)
        data_temp = dilate(data_temp,sphere)
    
        #In the 3d size matrix, assign the value of this diameter to everywhere that this sphere fit
        diams_3d[data_temp==1] = 2*r+1
        radii.append(r) #build the radius list
        if verbose:
                print("    Completed R = ",r,", which took ",tstring_h(time.time()-t)," ",np.count_nonzero(data_temp))
                #If there were no voxels left after the opening, then we're done, end the program
        if np.count_nonzero(data_temp) == 0:
            break

    if padding:
    	diams_3d = diams_3d[pad_val:-pad_val,pad_val:-pad_val,pad_val:-pad_val]
    	print("After removing",diams_3d.shape)

    #Now to find the actual number of voxels that belonged to each radius
    num_voxels = []
    for r in radii:
        num_voxels.append(np.count_nonzero(diams_3d == 2*r+1))

    return (radii, num_voxels, diams_3d)

def size_dist_basic2d(data,verbose=True):
    '''
    Data is a 2d matrixs of 1s and 0s, where 1s are the phase of interest
    '''
    
    maxR = np.max(np.shape(data))
    
    # Initialize output
    diams_2d = np.zeros(np.shape(data),dtype=int)
    radii = []
    
    for r in range(1,maxR+1):
        t = time.time()
        sphere = KernelCircle2d(r)
        
        #Method: erode then dilate (the definition of opening) using custom fft-convolution method
        data_temp = erode(data,sphere)
        data_temp = dilate(data_temp,sphere)
    
        #In the 2d size matrix, assign the value of this diameter to everywhere that this sphere fit
        diams_2d[data_temp==1] = 2*r+1
        radii.append(r) #build the radius list
        if verbose:
                print("    Completed R = ",r,", which took ",tstring_h(time.time()-t)," ",np.count_nonzero(data_temp))
                #If there were no voxels left after the opening, then we're done, end the program
        if np.count_nonzero(data_temp) == 0:
            break
    
    #Now to find the actual number of voxels that belonged to each radius
    num_voxels = []
    for r in radii:
        num_voxels.append(np.count_nonzero(diams_2d == 2*r+1))

    return (radii, num_voxels, diams_2d)

def get_davg_3phases(data,volumetric=True,verbose=False, saveit=False, phase='a', stdevs=False, return_davg_vs_axis=None):
	"""
	[davg1, davg2, davg3] = get_davg_3phases(data,volumetric=True,verbose=True,saveit=False, phase='a', stdevs=False, return_davg_vs_axis=None)
	For 3-phase voxelated data (3d array of 0s/1s/2s or 1s/2s/3s), finds the average
	particle size for each phase, in voxel units, and returns them in a list.
	If volumetric=True, does inscribed sphere and returns the volumetric average.
	If volumetric=False, will do grain assignment and return number-weighted d_avg

	Phase - can specify just 1 phase ID (integer), or 'a' does all 3.
		Note that 'a' ignores all negative phaseID values, since those are
		often used to mark erroneous voxels. If you want the size dist of a
		phase with a negative phaseID, you must manually specify e.g. phase=-1
		and do it one phase at a time.

	If saveit='path/to/filebasename', this will save diams_3d (the 3d map of diameters) as an npy file
	for each phase: path/to/filebasename_ph0.npy, ..._ph1.npy, etc.
	Currently only supports npy!


	stdevs: if True, will also return the PSD stdevs. The davgs and stdevs
	will come in their own lists packed into a tuple, like so:
		([davg1, davg2, davg3], [stdev1, stdev2, stdev3]) = get_davg_3phases(...)

	return_davg_vs_axis: Currently only works in Volumetric mode!! If you try
		to use this in volumetric=False mode, the davg_vs_axis will be returned,
		but will just be an empty list.
		None (default): don't do any of this or return this value. Otherwise...
		If set to an integer 0, 1, or 2, will also return a 1D
		numpy vector (for each phase requested) of the davg versus that axis,
		e.g. if axis=2 it will be the 3rd dim, usually called the Z direction.
		The method for this is by using the 3D array of inscribed sphere
		diameters in the space of the given phase,
		already generated to find davg, but instead of averaging it over the
		whole volume, it takes the average value within each plane of voxels
		along that axis. This way it can return Davg vs Z, without running into
		the finite size effect (it would be impossible to run inscribed sphere
		on a single plane of voxels, and unhelpful to run inscribed circle
		on a plane of voxels)

		They will be returned in their own list packed into a tuple like this:
		([davg1, davg2, davg3], [stdev1, stdev2, stdev3], [d_by_z1, d_by_z2, d_by_z3]) = get_davg_3phases(..., return_davg_vs_axis=2)
		Or if stdev=false,
		([davg1, davg2, davg3],  [d_by_z1, d_by_z2, d_by_z3]) = get_davg_3phases(..., return_davg_vs_axis=2)

		If set to None, they will not be returned, i.e.
		([davg1, davg2, davg3], [stdev1, stdev2, stdev3]) = get_davg_3phases(..., return_davg_vs_axis=None)

	"""
	#WARNING: Be very careful when modifying this function.
	#The integrated degradation model depends on this function!
	#Basically as long as volumetric=True and saveit=False and phase='a' (defaults),
	#and stdev=False, and return_davg_vs_axis=None (also defaults),
	#this func should return vol-weighted [davg1, davg2, davg3] and do nothing else.

	davg = []
	stdev = []
	davg_vs_axis = []

	if phase=='a':
		phases = [x for x in np.unique(data) if x>=0]
		if len(phases) != 3:
			raise ValueError('Input data has '+str(np.unique(data).size)+' phases, needs to be 3 only.')
	else:
		phases = [phase]

	for p in phases:
		t = time.time()
		phdata = data==p
		radii, num_voxels, diams_3d = size_dist_basic(phdata,verbose=verbose)
		if saveit:
			f_out = saveit+'_voldiams_ph'+str(p)+'.npy'
			if verbose:
				print('Saving ph',p,'diams_3d array as '+f_out)
			np.save(f_out, diams_3d)
		if volumetric:
			davg.append(np.mean(diams_3d[diams_3d>0]))
			stdev.append(np.std(diams_3d[diams_3d>0]))
			
	if stdevs:
		if return_davg_vs_axis is not None:
			return (davg, stdev, davg_vs_axis)
		else:
			return (davg, stdev)
	else:
		if return_davg_vs_axis is not None:
			return (davg, davg_vs_axis)
		else:
			return davg
