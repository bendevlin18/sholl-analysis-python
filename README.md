# sholl-analysis-python
Quick Script for performing sholl analysis on individual cells using python.

First, run the main.py script which will loop through a given folder and ask you to click on the center of each cell image so that it can run the calculations. Once the output csv files are created (1 per image) you can run the process_output.py function to combine all excel files into one output for analysis and stats.

Once you have isolated individual cells in .tif format, you can run this script to do the automatic skeletonization and ring generation. I usually use a simple segmentation in Ilastik for turning the cell image to a binary segmentation (black and white). If needed, you can also do this by hand in fiji using the thresholding function.

*** TO CHANGE RING SIZES ***
At the top of the script, in line 29-39, there is a dictionary created that contains the parameters for the rings (start radius, step size, and end radius). You can change these at will to more accurately apply to your specific images.
NOTE: Ring sizes are denoted in pixel values and should be converted to biological units (i.e. microns) after finishing analysis. 
