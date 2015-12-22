'''
peak detection - Use raw peaks (without calculated features such as amplitude,
width, etc.) to detect peaks and train classifier
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.text import Text
import sys
import peak_detect_utils1 as pd_utils
import os.path
import imp
imp.reload(pd_utils)


#######Start - Variables to be edited by user #####
#data_folder = ('C:/Users/prave_000/Dropbox/Analysis/python_progs/'
#                'python_data_analysis/peak_detect/raw_data/')
#predicted_fn = ('C:/Users/prave_000/Dropbox/Analysis/python_progs/'
#                'python_data_analysis/peak_detect/raw_rnd_predicted_less.csv')

#fn_match = '*.txt' # file name matching pattern
##fn_extension = '.txt'
##set_col_names = True # if data columns don't have a name, set = filename

##DOWN_SAMPLE_FACTOR = 1  # factor by which to down_sample. Need not be exact divisor of


##START = 10000
##END = 110000

##PEAK_DISPLAY_PREWIN = 750
##PEAK_DISPLAY_POSTWIN = 750

##DISPLAY_WIN = 1500

# additional features to mark per accepted peak
##detect_features = ['threshold'] # add more to the list if needed.
#detect_features = [] # use when manually selecting peaks for verifying accuracy

# gaussian filter 
##FILTER_WIN_SIZE = 21
##FILTER_SIGMA = 10.0

# yrange - with respect to the mean. eg. yrange = [-30e-12, 10e-12]
##yrange = [-40e-12, 10e-12]

#Use "\\" for windows, / for mac
#path_separator = "/" 
#path_separator = "\\" 
#match_filename = "*.csv"

#######End - Variables to be edited by user #######
###################################################

##data_folder = pd_utils.select_folder('Choose directory containing text data files')
####data_folder = 'C:/Users/taneja/Work/Dravet/text_data/sipsc'

##data_folder = data_folder + '/'
##print 'Folder selected:', data_folder

##predicted_fn = pd_utils.get_path('*.*', 'Choose data file with predictions')
####predicted_fn = 'C:/Users/taneja/Work/Dravet/text_data/sipsc/results/predicted.csv'
##print 'loading data file', predicted_fn

##peak_analysis = pd.read_table(predicted_fn, delimiter = ',')

def plot_peaks(data_folder, peak_analysis, fn_extension, START, END, 
                set_col_names, DOWN_SAMPLE_FACTOR, FILTER_WIN_SIZE, 
                FILTER_SIGMA, DISPLAY_WIN, yrange, detect_features, 
                plot_full_traces):
    
                            
    selected = peak_analysis['selected']
    file_name = peak_analysis['file_name']
    file_names = file_name.unique()
    num_files = len(file_names)
    #print 'file_names', file_names
    peak_x = peak_analysis['peak_x']
    peak_y = peak_analysis['peak_y']
    
    # plot peaks in sequential order
    peak_num = 0
    feature_num = 0
    the_file_name = file_names[0]
    
    
    features_xy = []
    for the_feature in detect_features:   
        x0 = peak_analysis.loc[peak_num, [the_feature + '_x']]
        y0 = peak_analysis.loc[peak_num, [the_feature + '_y']]
        features_xy.append((x0, y0))
        
    
        
    num_peaks = len(peak_x)
    
    plt.ion()
    
    # plot peaks in sequential order
    peak_num = 0
    feature_num = 0
    
    peak_x0 = peak_x[peak_num]
    peak_y0 = peak_y[peak_num]
    selected0 = selected[peak_num]
    
    if plot_full_traces == True:
        peak_x0 = peak_x[file_name == the_file_name]
        peak_y0 = peak_y[file_name == the_file_name]
        selected0 = selected[file_name == the_file_name]
        
        features_xy = []
        for the_feature in detect_features:   
            x0 = peak_analysis.loc[peak_num, [the_feature + '_x']]
            y0 = peak_analysis.loc[peak_num, [the_feature + '_y']]
            
            x0 = peak_analysis[the_feature + '_x'][file_name == the_file_name]
            y0 = peak_analysis[the_feature + '_y'][file_name == the_file_name]
            features_xy.append((x0, y0))       
    
    #print peak_x0[0:5]
    #print peak_y0[0:5]
    #print features_xy
    sys.stdout.flush()
    fig = plt.figure()
    
    fn= ''.join([data_folder, file_name[peak_num], fn_extension])
    data1d = pd_utils.pt_load_data(fn, DOWN_SAMPLE_FACTOR, START, END, set_col_names = set_col_names) # raw
    #data1d_filtered = pd_utils.pt_gaussian_filter(data1d, FILTER_WIN_SIZE, FILTER_SIGMA)
    
    # make initial plot for selecting or rejecting peak
    #pd_utils.pt_display_make_plot(peak_x[peak_num], peak_y[peak_num], features_xy, selected[peak_num], file_name[peak_num], data1d, 
    #                       data1d_filtered, DISPLAY_WIN, yrange, 
    #                        plot_full_traces = plot_full_traces)
    
    pd_utils.pt_display_make_plot(peak_x0, peak_y0, features_xy, selected0, the_file_name, data1d, 
                           DISPLAY_WIN, yrange, 
                            plot_full_traces = plot_full_traces)
    
    # use a list so that values can be preserved and updated between function calls.
    peak_num = [0]
    feature_num = [0]
    file_num = [0]
    
    # connect call back function pt_on_peak_pick to event manager
    cid = fig.canvas.mpl_connect('pick_event', lambda event: 
            pd_utils.pt_display_on_peak_pick(event, fig, num_peaks, num_files, 
            peak_analysis, DOWN_SAMPLE_FACTOR, START, END, set_col_names, FILTER_WIN_SIZE, FILTER_SIGMA,
            data_folder, fn_extension, 
            DISPLAY_WIN, yrange, detect_features, peak_num, file_num,
            plot_full_traces = plot_full_traces))