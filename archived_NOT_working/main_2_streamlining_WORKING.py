
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

direc = '/Users/bendevlin/Desktop/completed microglia'
files = os.listdir(direc)

os.chdir(direc)


###
# create radius information for the rings based on user-inputed params
###

ring_params = {

'start_radius': 5,
'step_size' : 20,
'end_radius' : 350

}

rads = np.arange(ring_params['start_radius'], ring_params['end_radius'], ring_params['step_size'])

df = pd.DataFrame(index = rads)

## skeletonize and remove islands from binary image
for file in files:

    if '.tif' in file:

        #read in img
        img = cv2.imread(direc +'/' + file)

        #skeletonize img using scikit-img skeletonize fxn, lee's method same as Fiji
        skeleton = skeletonize(np.invert(img), method = 'lee')

        #process to remove small objects
        processed = morphology.remove_small_objects(skeleton.astype(bool), min_size=10, connectivity=25).astype(int)
        processed = np.where(processed > 0, 255, 1)

        #structuring the skeleton array to be the right shape
        new_arr = np.zeros(shape = np.shape(processed)[0:2])

        for i in range(len(processed)):
            for j in range(len(processed)):
                new_arr[i][j] = processed[i][j][1]


        new_arr = morphology.dilation(new_arr, morphology.disk(radius=1))


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
            arr = np.zeros(np.shape(processed)[0:2])
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
        ax1.imshow(new_arr, origin = 'lower')
        ax1.set_title('Look good??')
        ax2.set_title('Original Image')
        
        for circ in circles:
            ax1.imshow(circ, interpolation='nearest', origin='lower', cmap = cmap2)
        ax1.invert_yaxis()
        ax2.imshow(img)
        plt.show()


        intersects = []

        def calc_intersection(circ_arr, skeleton_arr):
            intersects = []

            for i in range(len(circ_arr)):
                for j in range(len(new_arr)):
                    if circ_arr[i][j] == skeleton_arr[i][j]:
                        intersects = np.append(intersects, [i, j])

            return(intersects)

        z = ['l'] * len(rads)
        intersections = []
        it = -1
        for circ in circles:
            it += 1
            z[it] = calc_intersection(circ, new_arr)
            intersections = np.append(intersections, len(calc_intersection(circ, new_arr))/2)
            

        it = -1
        pairs = []
        xs = []
        ys = []
        for iterator in z:
            it += 1
            pairs = zip(iterator[::2], iterator[1::2])
            for inter in list(pairs):
                xs = np.append(xs, inter[0])
                ys = np.append(ys, inter[1])

        x, y = xs, ys
        
        

    ### need to plot everything and save the batched output

        fig, ax1 = plt.subplots(1, figsize = (10,10), sharey = False)
        ax1.scatter(center[0][0], center[0][1])
        ax1.imshow(new_arr, origin = 'lower')
        ax1.set_title(file)
        
        for circ in circles:
            ax1.imshow(circ, interpolation='nearest', origin='lower', cmap = cmap2)
        ax1.invert_yaxis()
        ax1.scatter(y, x)
        plt.savefig('sholl_' + file + '.png')
        plt.show()
        alldone = plt.ginput(1)
        plt.close()

        df[file] = z

df.T.to_csv('raw_output.csv')
        