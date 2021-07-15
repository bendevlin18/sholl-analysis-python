
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import morphology
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage import draw, morphology
import matplotlib as mpl
from matplotlib.colors import colorConverter
from scipy.ndimage import generic_filter
from skimage.morphology import medial_axis
from tkinter import *
from tkinter import filedialog


root = Tk()
direc = filedialog.askdirectory(title = 'Select a Folder')
files = [i for i in os.listdir(direc) if '.tif' in i]

if not os.path.exists(os.path.join(direc, 'sholl_output')):
    os.mkdir(os.path.join(direc, 'sholl_output'))

output_direc = os.path.join(direc, 'sholl_output')
os.chdir(output_direc)


###
# create radius information for the rings based on user-inputed params
###

ring_params = {

'start_radius': 20,
'step_size' : 20,
'end_radius' : 350

}

rads = np.arange(ring_params['start_radius'], ring_params['end_radius'], ring_params['step_size'])

df = pd.DataFrame(index = rads)

## skeletonize and remove islands from binary image
for file in files:


    #read in img
    img = cv2.imread(direc +'/' + file)

    #skeletonize img using scikit-img skeletonize fxn, lee's method same as Fiji
    skeleton = skeletonize(np.invert(img), method = 'lee')

    #process to remove small objects
    processed = morphology.remove_small_objects(skeleton.astype(bool), min_size=10, connectivity=25).astype(int)
    processed = np.where(processed > 0, 255, 0)

    #structuring the skeleton array to be the right shape
    processed_skeleton = np.zeros(shape = np.shape(processed)[0:2])

    for i in range(len(processed)):
        for j in range(len(processed)):
            processed_skeleton[i][j] = processed[i][j][1]

    #plotting original image to get center pt
    plt.imshow(img)
    plt.title('Pls select center')
    center = plt.ginput(1)
    plt.close()

    ##making the circles from the radius parameters
    circles = ['1'] * len(rads)
    rr = []
    cc = []
    w = -1
    for rad in rads:
        w += 1
        arr = np.ones(np.shape(processed)[0:2])
        rr, cc = draw.circle_perimeter(int(center[0][1]), int(center[0][0]), radius=rad, shape=arr.shape)
        arr[rr, cc] = 255
        circles[w] = arr

    ## plot circles here to double check parameters adequately capture microglia skeleton

    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('black')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 0.8, cmap2.N+3)
    cmap2._lut[:,-1] = alphas

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize = (14,5), sharey = False)
    ax1.scatter(center[0][0], center[0][1])
    ax1.imshow(processed_skeleton, origin = 'lower')
    ax1.set_title('Look good??')
    ax2.set_title('Original Image')
    
    for circ in circles:
        ax1.imshow(circ, interpolation='nearest', origin='lower', cmap = cmap2)
    ax1.invert_yaxis()
    ax2.imshow(img)
    plt.show()

    ## making the skeleton thicc so that we don't miss any overlaps
    mgla_arr = morphology.dilation(processed_skeleton, morphology.disk(radius=1))

    def calc_intersection(circ_arr, skeleton_arr):
        intersects = []

        for i in range(len(circ_arr)):
            for j in range(len(mgla_arr)):
                if circ_arr[i][j] == skeleton_arr[i][j]:
                    intersects = np.append(intersects, [i, j])

        return(intersects)

    z = ['l'] * len(rads)
    intersections = []
    it = -1
    for circ in circles:
        it += 1
        z[it] = calc_intersection(circ, mgla_arr)
        intersections = np.append(intersections, len(calc_intersection(circ, mgla_arr))/2)
        

    
## defining the distance formula so that we can calculate distance for potentially overlapping points

    def dist_formula(x1, y1, x2, y2):

        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return d


## function for separating an array of numbers into x, y coordinates

    def x_y_separate(arr):
        
        xs = []
        ys = []

        pairs = zip(arr[::2], arr[1::2])
        for inter in list(pairs):
            xs = np.append(xs, inter[0])
            ys = np.append(ys, inter[1])

        x, y = xs, ys
        
        return x, y


    ## this is cleaning the intersections and saving the intersections
    ## we will do this individually for each image
    ## input is 'z' which is the output from the current calc_intersections fxn

    total_intersections = pd.DataFrame(z, index = rads)

    print(total_intersections)
    intersections_to_plot = pd.DataFrame()

    for idx, row in total_intersections.iterrows():

        vals = row.dropna().values
        x, y = x_y_separate(vals)   
        
        cleaned_intersections = pd.DataFrame(pd.DataFrame([x, y]).T)

        for cur in cleaned_intersections.iterrows():
            for new in cleaned_intersections.iterrows():
                if dist_formula(cur[1][0], cur[1][1], new[1][0], new[1][1]) < 10:
                ## and now to set the new index values equal to the current index values
                    cleaned_intersections.loc[new[0]] = cleaned_intersections.loc[cur[0]]
        
        ## dropping the duplicates
        final_intersections = cleaned_intersections.drop_duplicates()
        final_intersections['ring'] = idx
        final_intersections.set_index('ring', inplace = True)
        
        ## saving the intersection x and y values to be plotted
        intersections_to_plot = intersections_to_plot.append(final_intersections)

    ## filter for finding endpoints (point flanked by only 1 other point)
    def num_endpoints(p):
        return 255 * ((p[4]==255) and np.sum(p)==510)

    endpoint_arr = generic_filter(processed_skeleton, num_endpoints, (3, 3))

    endpoints = np.unique(endpoint_arr, return_counts = True)[1][1]
    intersections_to_plot['endpoints'] = [endpoints] * len(intersections_to_plot)
    intersections_to_plot.to_csv(os.path.join(output_direc, file[0:-4] + '_raw_intersections.csv'))

    ep = []

    for i in range(len(endpoint_arr)):
        for j in range(len(endpoint_arr)):
            if endpoint_arr[i][j] == 255.0:
                ep = np.append(ep, [i, j])

    x_ep, y_ep = x_y_separate(ep)   

    ### need to plot everything
    fig, ax1 = plt.subplots(1, figsize = (10,10), sharey = False)
    ax1.scatter(center[0][0], center[0][1])
    ax1.imshow(processed_skeleton, origin = 'lower')
    ax1.set_title(file)
    
    for circ in circles:
        ax1.imshow(circ, interpolation='nearest', origin='lower', cmap = cmap2)

    ax1.invert_yaxis()
    ax1.scatter(intersections_to_plot[1].values, intersections_to_plot[0].values)
    ax1.scatter(y_ep, x_ep, marker = '^')
    plt.savefig('sholl_' + file + '.png')
    plt.show()
    alldone = plt.ginput(1)
    plt.close()