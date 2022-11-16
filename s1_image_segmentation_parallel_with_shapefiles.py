# This script does the following:
#
# 1. Steps through all of the shapefiles associated with the aerial photos and
# determines whether the shapefile falls entirely within in the fjord. Photos
# that don't fall entirely within the fjord are excluded from analysis. A file
# titled 'input_files.npy' is created that contains the list of photos to be 
# processed.
#
# 2. Steps through all of the photos in the 'input_files.npy' file and
# performs image segmentation. For each it produces a npz file that contains
# the total number of pixels (npixels), an array with the area of each
# individual iceberg (icebergArea), the pixel size (pixArea), and the edges
# of the icebergs (edges) for plotting purposes. It also produces a figure 
# to show the results of the image segmentation.


import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import cv2

import glob
import os

from scipy.ndimage.filters import gaussian_filter

import multiprocessing as mp
import timeit

import shapefile

from shapely.geometry import Polygon, Point

plt.ioff()
#%%


def findIcebergs(input_file):    
    # https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/
    
    
    print('Processing image ' + os.path.basename(input_file))
    
    img = cv2.imread(input_file)
    # use value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, img = cv2.split(hsv)

    # smooth to remove small icebergs and whitecaps
    img = cv2.GaussianBlur(img,(11,11),0)
    
    npixels = img.shape[0]*img.shape[1] # store total number of pixels; later store number of pixels in each iceberg
    aspect_ratio = float(img.shape[0])/img.shape[1]
    
    with open(input_file[:-3] + 'jgw') as f:
        pixArea = f.readline().rstrip() # find area of pixel for each image
        pixArea = (np.array(pixArea,dtype='float32'))**2
    f.close()

   
    if np.max(img) > 200: # image contains icebergs!
        # enhance contrast by stretching the image PROBLEM IF NO ICEBERGS!!
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype('uint8')
                
        # add border to image to detect icebergs along the boundary
        bordersize=1
        img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, 
                                   left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=0 )
        
        # threshold image
        _, img_ = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
        #_, img_ = cv2.threshold(img,40,255, cv2.THRESH_BINARY)
            
        
        ## segment using cluster detection, using canny edge detection
        # find edges
        edges = cv2.dilate(cv2.Canny(img_,0,255),None)
        
        # detect contours: only external contours, and keep all points along contours
        _, contours, _ = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # remove holes; this command no longer seems necessary?
        #contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] < 0]
        
        # fill contours on original image
        cv2.drawContours(img, contours, contourIdx=-1, color=(255,255,255),thickness=-1).astype('float32')
        
        
    
        # calculate area of icebergs
        icebergArea = np.zeros(len(contours))
        for k in range(len(icebergArea)):
            icebergArea[k] = cv2.contourArea(contours[k])
    
    else: # create empty arrays if no icebergs are found
        icebergArea = []    
        edges = []
    
    photo_dir = os.path.dirname(input_file)
    processed_directory = photo_dir + '/processed/'
    
    np.savez(processed_directory + '/' +  os.path.basename(input_file)[:-4] + '_icebergs.npz', npixels=npixels, icebergArea=icebergArea, pixArea=pixArea, edges=edges)

    plt.figure(figsize=(15,15*aspect_ratio))
    ax = plt.axes([0,0,1,1])
    img = cv2.imread(input_file)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(RGB_img)
    
    if np.sum(icebergArea)>0: # if contains icebergs, plot the edges using a map of the edges (instead of the vectors)
        edges_plot = edges[1:-1,1:-1].astype('float32')
        edges_plot[edges_plot==0] = np.nan
        ax.imshow(edges_plot)
    
    ax.axis('off')
    plt.savefig(processed_directory + '/' + os.path.basename(input_file)[:-4] + '_icebergs.jpg',format='jpg',dpi=300)
    plt.close()
    
#%%    
def main():
    
    
    fjord = shapefile.Reader('./data/fjord_outline_copy.shp')
    fjord = fjord.shapeRecords()[0]
    fjord_polygon = Polygon(fjord.shape.points)
  
    campaign_shapefiles = sorted(glob.glob('../HARBORSEAL_' + year + '/footprints/*.shp')) 
    
   
    input_files_all = [] # input files for all campaigns during the year
    
    for j in np.arange(0,len(campaign_shapefiles)):
        campaign = campaign_shapefiles[j][-22:-14]
        
        photo_dir = glob.glob('../HARBORSEAL_' + year + '/JHI/' + campaign + '/*' + campaign + '_photo/')[0]
        processed_directory = photo_dir + 'processed/'
        
        if not os.path.isdir(processed_directory):
            os.mkdir(processed_directory)
        
        # footprints of all photos from that campaign
        footprints = shapefile.Reader(campaign_shapefiles[j])
       
        
        if os.path.exists(photo_dir + '/input_files.npy'):
            input_files = np.load(photo_dir + '/input_files.npy').tolist()
            
        else:           
            
            input_files = []
            
            for k in np.arange(0,footprints.numRecords):    
                photo = footprints.shapeRecords()[k] # extract footprint from individual photo
                
                if int(year)<2018:
                    photo_name = photo.record[1] # extract name of individual photo
                else:
                    photo_name = photo.record[0] # extract name of individual photo
                
                # seems to be different format for photo.record for older photos
                
                photo_polygon = Polygon(photo.shape.points) # get polygon boundary
        
                infjord = fjord_polygon.contains(photo_polygon) # is photo completely within the fjord?
                
                if infjord==True:
                    input_files = input_files + [photo_dir + photo_name]
                    
            np.save(photo_dir + '/input_files.npy', input_files)
        
        input_files_all = input_files_all + input_files
        
    print('Number of files to process: ' + str(len(input_files_all)))
    
    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(findIcebergs, input_files_all)  
     
    
#%%    
if __name__ == '__main__':    
        
    start_time = timeit.default_timer()
    
    year = '2008' # only processes one year at a time
    
    main()
    
    elapsed = timeit.default_timer() - start_time

    print('Elapsed time: ' + str(elapsed/60) + ' min')
    



