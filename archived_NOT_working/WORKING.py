
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

direc = 'C:\\Users\\Ben\\Dropbox\\bilbo_lab_spr2020\\il34_project\\sample_data_3dmorph\\ind_mgla'
list_of_files = os.listdir(direc)

files = []
for name in list_of_files:
    if '.tif' in name:
        files = np.append(files, name)

print(files)

os.mkdir(direc + '\\sholl_output')
rad_1 = 20
rad_2 = 30
rad_3 = 40
rad_4 = 50
rad_5 = 60
rad_6 = 70
rad_7 = 80
rad_8 = 90
rad_9 = 100
rad_10 = 110
rad_11 = 120
rad_12 = 130
rad_13 = 140
rad_14 = 150
rad_15 = 160
rad_16 = 170
rad_17 = 180
rad_18 = 190
rad_19 = 200
rad_20 = 210

rads = [rad_1, rad_2, rad_3, rad_4, rad_5, rad_6, rad_7, rad_8, rad_9, rad_10, rad_11, rad_12, rad_13, rad_14, rad_15, rad_16, rad_17, rad_18, rad_19, rad_20]


df = pd.DataFrame(index = rads)

## cleaning function for images that already have the brightness bumppped (usually hit the auto button 3 times or so for 8bit images)
for file in files:
    img = cv2.imread(direc +'\\' + file)

    ## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    ## detect where most of the background is coming from so you can get that out
    num = np.histogram(dst.flatten(), bins = 50)

    ## gets the value where most background is, so that we can subtract it away
    bg_val = num[1][np.argmax(num[0]) + 6]

    new = np.where(dst < bg_val, 0, dst)
    skeleton = skeletonize(new)

    processed = morphology.remove_small_objects(skeleton.astype(bool), min_size=50, connectivity=25).astype(int)
    processed = np.where(processed > 0, 255, 0)

    new_arr = np.zeros(shape = (500, 500))

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

    
    plt.imshow(new)
    center = plt.ginput(1)
    plt.close()


    circles = ['1'] * len(rads)
    rr = []
    cc = []
    w = -1
    for rad in rads:
        w += 1
        arr = np.zeros((500, 500))
        rr, cc = draw.circle_perimeter(int(center[0][1]), int(center[0][0]), radius=rad, shape=arr.shape)
        arr[rr, cc] = 255
        circles[w] = arr

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
    for arr in circles:
        it += 1
        z[it] = calc_intersection(arr)[0]
        intersections = np.append(intersections, calc_intersection(arr)[1])

    x, y = intersections.reshape( int(len(intersections)/2) , 2).T
    
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('black')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 0.8, cmap2.N+3)
    cmap2._lut[:,-1] = alphas
    
    
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize = (15, 30))

    table_vals = list(zip(rads, z))
    col_labels = ['dist_from_soma', '# intersections']
    row_labels = rads
    ax2.table(cellText=table_vals, colWidths = [.2]*3, colLabels=col_labels, cellLoc = 'center', loc = 16, fontsize = 11).scale(1, 4)
    ax1.imshow(new_arr, origin = 'lower')
    ax1.imshow(circles[0], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[1], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[2], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[3], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[4], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[5], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[6], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[7], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[8], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.imshow(circles[9], interpolation='nearest', cmap=cmap2, origin='lower')
    ax1.scatter(center[0][0], center[0][1])
    ax1.scatter(y, x) 
    ax1.set_title('Filename: ' + file)
    ax2.imshow(new, origin = 'lower')
    plt.savefig(direc + '\\sholl_output\\' + 'output_image_' + file + '.png')
    plt.show()
    df[file] = z

df.T.to_csv(direc + '\\sholl_output\\' + 'output.csv')