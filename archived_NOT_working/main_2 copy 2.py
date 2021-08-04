
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import morphology
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage import draw
import matplotlib as mpl
from matplotlib.colors import colorConverter

direc = '/Users/bendevlin/Desktop/images'
files = os.listdir(direc)

rad_1 = 10
rad_2 = 20
rad_3 = 35
rad_4 = 50
rad_5 = 75
rad_6 = 100
rad_7 = 115
rad_8 = 130
rad_9 = 150
rad_10 = 175

rads = [rad_1, rad_2, rad_3, rad_4, rad_5, rad_6, rad_7, rad_8, rad_9, rad_10]


df = pd.DataFrame(index = rads)

## cleaning function for images that are already masked and processed
for file in files:
    img = cv2.imread(direc +'/' + file)

    skeleton = skeletonize(np.invert(img), method = 'lee')


    processed = morphology.remove_small_objects(skeleton.astype(bool), min_size=10, connectivity=25).astype(int)
    processed = np.where(processed > 0, 0, 255)

    print(np.shape(processed)[0:2])

    new_arr = np.zeros(shape = np.shape(processed)[0:2])

    for i in range(len(processed)):
        for j in range(len(processed)):
            new_arr[i][j] = processed[i][j][1]

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 18))
    # ax1.imshow(new)
    # ax1.set_title('Smoothed Image')
    # ax2.imshow(skeleton)
    # ax2.set_title('Skeleton Image')
    # ax3.imshow(new_arr)
    # ax3.set_title('Cleaned Final Skeleton ' + file)
    # plt.show()

    
    plt.imshow(img)
    center = plt.ginput(1)
    plt.close()


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
2

    intersects = []

    def calc_intersection(arr):
        intersects = []

        for i in range(len(arr)):
            for j in range(len(new_arr)):
                if arr[i][j] == 255.0:
                    if arr[i][j] == new_arr[i][j]:
                        intersects = np.append(intersects, [i, j])
                    elif arr[i][j] != new_arr[i][j]:
                        pass
                elif arr[i][j] != 255.0:
                    pass
        print(len(intersects) / 2)
        return(len(intersects) / 2, intersects)




    z = ['l'] * len(rads)
    intersections = []
    it = -1
    for circ in circles:
        it += 1
        z[it] = calc_intersection(circ)[0]
        intersections = np.append(intersections, calc_intersection(circ)[1])

    x, y = intersections.reshape( int(len(intersections)/2) , 2).T
    
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('black')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 0.8, cmap2.N+3)
    cmap2._lut[:,-1] = alphas
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 18))
    
    plt.figure(figsize = (10, 8))
    table_vals = list(zip(rads, z))
    col_labels = ['dist_from_soma', '# intersections']
    row_labels = rads
    plt.table(cellText=table_vals, colWidths = [0.1]*3, rowLabels=row_labels, colLabels=col_labels, cellLoc = 'center', loc = 14)
    plt.imshow(new_arr, origin = 'lower')
    plt.imshow(circles[0], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[1], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[2], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[3], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[4], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[5], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[6], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[7], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[8], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.imshow(circles[9], interpolation='nearest', cmap=cmap2, origin='lower')
    plt.scatter(center[0][0], center[0][1])
    plt.scatter(y, x) 
    plt.title('Sholl Analysis ' + file)
    plt.savefig(direc + '/' + 'output_image_' + file + '.png')
    plt.show()
    df[file] = z

df.T.to_csv(os.path.join(direc, 'output.csv'))