

import os
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import filedialog


root = Tk()
direc = filedialog.askdirectory(title = 'Select a Folder')
os.chdir(direc)

## making a list of only the relevant files
files = [file for file in os.listdir(direc) if '.csv' in file]

## change these to make sure that all your data for all rings is extracted appropriately!
ring_params = {

'start_radius': 15,
'step_size' : 10,
'end_radius' : 200

}

rads = np.arange(ring_params['start_radius'], ring_params['end_radius'], ring_params['step_size'])
final_df = pd.DataFrame(index = rads)

## initializing an empty dataframe and dict to store the data
#final_df = pd.DataFrame(index = np.unique(pd.read_csv(os.path.join(direc, files[0]), index_col = 0).index))
endpoints = {}

## iterating through each file in the list
for file in files:
    
    ## importing the dataframe
    df = pd.read_csv(os.path.join(direc, file), index_col = 0)
    
    ## extracting number of endpoints for the ramification calculation
    endpoints[file] = df['endpoints'].values[0]
    
    ## extracting intersection count at each ring step
    ## and saving it to the output dataframe
    int_df = pd.DataFrame(np.unique(df.index, return_counts = True)).T.set_index(0).rename(columns = {1: file})
    final_df = final_df.join(int_df)
    print(final_df)
    
epts_df = pd.DataFrame(endpoints, index = [0]).T.rename(columns = {0: 'endpoints'})

## joining the endpoint information to the intersection information
## calculating the ramification index
output_df = final_df.T.join(epts_df)
output_df['ramification_index'] = output_df['endpoints'] / output_df[output_df.columns[0]]

## saving the file
output_df.to_csv('combined_output.csv')

print('All finished! You can find the output here: ', direc + '/combined_output.csv')