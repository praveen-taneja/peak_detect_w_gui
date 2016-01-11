'''
generate training data for peak detection using machine learning
'''

#import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
#import collections
import peak_detect_utils_w_gui as pd_utils
import os.path
import imp
imp.reload(pd_utils)


#######Start - Variables to be edited by user #####
#Use "\\" for windows, / for mac

#data_folder = ('/Users/taneja/work/dravet_spont_and_mini_epsc_ipsc/text_data/sepsc/')
                
#fn_match = '*.csv' # file name matching pattern
#fn_extension = '.csv'
#set_col_names = True # if data columns don't have a name, set = filename

# a point is a possible peak if it is higher than PEAK_NPNTS_THRESH number of
# points on either side.
#DOWN_SAMPLE_FACTOR = 1  # factor by which to down_sample. Need not be exact divisor of
# length
#PEAK_NPNTS_THRESH = 1000
#PEAK_POLARITY = -1 

# additional features to mark per accepted peak
#detect_features = ['threshold'] # add more to the list if needed.

# start and end indices between which to detect peaks.
#START = 0
#END = 1014999

# gaussian filter 
#FILTER_WIN_SIZE = 21
#FILTER_SIGMA = 10.0

# number of points to crop before and after peak for training classifier
#PEAK_PRE_CROP_WIN = 80
#PEAK_POST_CROP_WIN = 50

# number of points to display while selecting peaks for training - xrange
#DISPLAY_WIN = 10000 # 

# yrange - with respect to the mean. eg. yrange = [-30e-12, 10e-12]
#yrange = [-2, 2]

#######End - Variables to be edited by user #######
###################################################

#data_folder = pd_utils.select_folder('Choose directory containing text data files')
#data_folder = data_folder + '/'
#print 'Folder selected:', data_folder

#results_dir = data_folder + 'results/'
#if os.path.exists(results_dir) == False:
#    os.makedirs(results_dir)

#parent_dir = os.path.abspath(os.path.join(data_folder, os.pardir))

def generate_examples(data_folder, results_dir, fn_match, fn_extension, START, 
                        END, set_col_names, DOWN_SAMPLE_FACTOR,
                        FILTER_WIN_SIZE, FILTER_SIGMA, PEAK_POLARITY,
                        PEAK_NPNTS_THRESH, DERIV2_THRESH, display_deriv2, PEAK_PRE_CROP_WIN,
                        PEAK_POST_CROP_WIN,DISPLAY_WIN, yrange,
                        detect_features):
    fn_list = glob.glob(data_folder + fn_match)
    print ' '
    print 'Looking for files in', data_folder
    print 'Found', len(fn_list), 'files'
    print ' '

    new_analysis = True
    if os.path.exists(results_dir+'cropped_peaks.csv') and os.path.exists(results_dir+'peak_vars.csv'):
        # check if number of rows is same. 
        print 'cropped peaks.csv and peak_vars.csv already exist'
        #raw_input() # ToDo
        new_analysis = False
        
        peak_analysis = pd.read_table(results_dir+'peak_vars.csv', delimiter = ',')
        peak_x = peak_analysis['peak_x']
        peak_y = peak_analysis['peak_y']
        file_name = peak_analysis['file_name']

    if new_analysis:
    # returns only peak_x, peak_y, file_name, selected to save memory
        peak_analysis = pd_utils.pt_get_all_peaks(fn_list, results_dir, START, END, set_col_names, DOWN_SAMPLE_FACTOR,
                        FILTER_WIN_SIZE, FILTER_SIGMA, PEAK_POLARITY,
                        PEAK_NPNTS_THRESH, DERIV2_THRESH, 
                        PEAK_PRE_CROP_WIN, PEAK_POST_CROP_WIN,
                        detect_features)
            
    #print 'peak_analysis.shape =', peak_analysis.shape

    peak_x = peak_analysis['peak_x']
    peak_y = peak_analysis['peak_y']
    file_name = peak_analysis['file_name']
    

    num_peaks = len(peak_x)

    plt.ion()

    # detect peaks in random order so that we get a good sampling of all peaks
    # random sequence of unique peak numbers. 
    
    # TO DO: DON'T RE-ANALYZE PEAKS WHICH WERE ALREADY ANALYZED IN CASE WE ARE 
    # USING OLDER ANALYSIS. IF NUM PEAKS IS LARGE, PROBABILITY OF THIS HAPPENING 
    # IS LOW ANYWAYS. 
    peak_nums_rnd = random.sample(range(0, num_peaks), num_peaks)
    
    peak_num = 0
    feature_num = 0

    fig = plt.figure()
    # index of a random peak
    peak_num_rnd = peak_nums_rnd[peak_num]
    
    #print 'data_folder, file_name[peak_num_rnd], fn_extension', data_folder,peak_num_rnd,  file_name[peak_num_rnd], fn_extension
    fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
    data1d = pd_utils.pt_load_data(fn, DOWN_SAMPLE_FACTOR, START, END, set_col_names = set_col_names) # raw
    #data1d_filtered = pd_utils.pt_gaussian_filter(data1d, FILTER_WIN_SIZE, FILTER_SIGMA)
    deriv2_filt = pd_utils.pt_filtered_deriv2(data1d, FILTER_WIN_SIZE, FILTER_SIGMA)
    # make initial plot for selecting or rejecting peak
    pd_utils.pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], data1d, 
                            deriv2_filt, display_deriv2, DISPLAY_WIN, yrange, pick_feature = None)

    # use a list so that values can be preserved and updated between function calls.
    peak_num = [0]
    feature_num = [0]

    # connect call back function pt_on_peak_pick to event manager
    cid = fig.canvas.mpl_connect('pick_event', lambda event: 
            pd_utils.pt_on_peak_pick(event, fig, peak_nums_rnd, num_peaks, 
            peak_analysis, DOWN_SAMPLE_FACTOR, START, END, set_col_names, FILTER_WIN_SIZE, FILTER_SIGMA,
            PEAK_PRE_CROP_WIN, results_dir, data_folder, fn_extension, 
            DISPLAY_WIN, yrange, detect_features, peak_num, feature_num, display_deriv2))
