# This script steps through all of the npz files for each campaign and
# concatenates them into one campaign file in order to later calculate iceberg
# size distributions.

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime as dt

import cv2

import glob
import os

from scipy.ndimage.filters import gaussian_filter

import multiprocessing as mp
import timeit

#%%
campaign_dirs = sorted(glob.glob('../HARBORSEAL_20*/JHI/20*'))

campaigns = [os.path.basename(x) for x in campaign_dirs] # list of campaigns for saving purposes


for j in range(0,len(campaign_dirs)):
    files = sorted(glob.glob(campaign_dirs[j] + '/*_photo/processed/*npz'))
    if len(files)==0:
        continue
        

    surveyArea_total = 0

    icebergArea = np.array([])
    
    # step through all photos for each campaign
    for k in range(0,len(files)):
        data = np.load(files[k])
    
        if len(data['icebergArea'])>0:            
            icebergArea = np.concatenate((icebergArea,data['icebergArea']*data['pixArea'])) # in m^2
                    
        surveyArea = data['npixels']*data['pixArea'] # in m^2
            
        surveyArea_total += surveyArea

    np.savez('iceberg_areas/' + campaigns[j] + '.npz' , icebergArea=icebergArea, surveyArea=surveyArea_total )