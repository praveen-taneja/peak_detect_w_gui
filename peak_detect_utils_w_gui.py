import numpy as np
import sys
import os
import matplotlib.pylab as plt
import math
import pandas as pd
from scipy import stats
import scipy.signal as signal # for correlation
from matplotlib.text import Text
from matplotlib.lines import Line2D


import wx

    
def select_folder(title_text):
    """
    http://www.blog.pythonlibrary.org/2010/06/26/the-dialogs-of-wxpython-part-1-of-2/
    Show the DirDialog and print the user's choice to stdout
    """
    app = wx.App()
    dlg = wx.DirDialog(None, title_text,
                       style=wx.DD_DEFAULT_STYLE
                       #| wx.DD_DIR_MUST_EXIST
                       #| wx.DD_CHANGE_DIR
                       )
    if dlg.ShowModal() == wx.ID_OK:
        #print "You chose %s" % dlg.GetPath()
        folder_path = dlg.GetPath()
    else:
        sys.exit()
    dlg.Destroy()
    return folder_path
    
def get_path(wildcard, title_text):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, title_text, wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
        sys.exit()
    dialog.Destroy()
    return path


def pt_downsample(df, factor):
    #http://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
    data1d = df.values
    col_name = df.columns.values[0]
    data1d = data1d.reshape([len(data1d), ])
    pad_size = math.ceil((float(len(data1d))/factor))*factor - len(data1d)
    data1d_padded = np.append(data1d, np.zeros(pad_size)*np.NaN)
    data1d = stats.nanmean(data1d_padded.reshape(-1, factor), axis = 1)
    print type(data1d), col_name
    data1d = pd.DataFrame(data1d, columns = [col_name])
    return data1d
    

def pt_load_data(filename, down_sample_factor, start, end, set_col_names = False):
    if set_col_names:
        fn_base = os.path.basename(filename)
        fn_no_ext = os.path.splitext(fn_base)[0]
        df = pd.read_table(filename, delimiter = ',', header=None, names =
                            [fn_no_ext])
    else:
        df = pd.read_table(filename, delimiter = ',')
    if down_sample_factor > 1:
        df = pt_downsample(df, down_sample_factor)
     # 02/03/16 start index at 0 instead of 'start'                
    df = df[start:end].reset_index(drop = True)
    return df

def pt_get_all_peaks(fn_list, results_dir, start, end, set_col_names, 
                    down_sample_factor, filter_win_size, filter_sigma, 
                    PEAK_POLARITY, PEAK_NPNTS_THRESH, DERIV2_THRESH, 
                    PEAK_PRE_CROP_WIN, PEAK_POST_CROP_WIN, detect_features):
                    


    ''' Return a data frame with all the peaks including false positives. 
        # A point is a possible peak if it is higher than PEAK_NPNTS_THRESH 
        number of
    '''
    peak_analysis = pd.DataFrame()
    #cropped_peaks_all = pd.DataFrame()
    col_num = 0
    
    # find peaks
    peak_num = 0
    new_file = True
    if os.path.exists(results_dir+'cropped_peaks.csv'):
        os.remove(results_dir+'cropped_peaks.csv')
        
    num_files = len(fn_list)
    for filenum, filename in enumerate(fn_list):
        data1d = pt_load_data(filename, down_sample_factor, start, end, set_col_names = 
                                set_col_names) # raw 
        col_name = data1d.columns.values[0]
        ##$##data1d = pt_gaussian_filter(data1d, filter_win_size, filter_sigma) # filtered
        
        #print 'PEAK_NPNTS_THRESH, PEAK_POLARITY', PEAK_NPNTS_THRESH, PEAK_POLARITY
        #print len(deriv2_filt), PEAK_NPNTS_THRESH, 5000, PEAK_POLARITY
        #sys.stdout.flush()
        #plt.plot(deriv2)
        #plt.plot(deriv2_filt)
        #plt.show()
        print 'file', filenum + 1, 'of', num_files, ':',os.path.basename(filename)
        peak_x = pt_local_extrema_w_thresh(data1d, filter_win_size, 
                                            filter_sigma, PEAK_POLARITY, 
                                            PEAK_NPNTS_THRESH, DERIV2_THRESH)
                                
        #print 'type(peak_x)', type(peak_x)
        '''
        if PEAK_POLARITY == 1: # minima in double derivative
            
            ##$##peak_x = signal.argrelextrema(data1d, np.greater, order = 
            ##$##                            PEAK_NPNTS_THRESH)
            #peak_x = signal.argrelextrema(pd.rolling_std(data1d, 50, center=True), np.greater, order = 
            #                            PEAK_NPNTS_THRESH)
        else: # maxima in double derivative
            ##$##peak_x = signal.argrelextrema(data1d, np.less, order = 
            ##$##                            PEAK_NPNTS_THRESH)
        '''
        # ignore peaks that occur within PEAK_POST_CROP_WIN from the start or
        # PEAK_POST_CROP_WIN from the end as we cannot crop a peak fully for 
        # classifier training in such cases.
        ##$##peak_x = peak_x[0]
        ###print type(peak_x), peak_x[0:20]
        peak_x = peak_x[peak_x > PEAK_PRE_CROP_WIN]
        peak_x = peak_x[peak_x < (len(data1d) - PEAK_POST_CROP_WIN)]
        num_peaks = len(peak_x)

        #print 'file', filenum + 1, 'of', num_files, '(',os.path.basename(filename),')', ': num peaks (including false-positives)', len(peak_x)
        #print ' '
        # make empty data frame for cropped peaks
        x = np.zeros([num_peaks, PEAK_POST_CROP_WIN + PEAK_PRE_CROP_WIN])
        x[:] = np.NAN
        cropped_peaks = pd.DataFrame(x)
        peak_num = 0
        
        for this_peak in peak_x:
            peak_crop = data1d[this_peak - PEAK_PRE_CROP_WIN : this_peak + 
                                PEAK_POST_CROP_WIN]
            #print 'peak_crop.shape', peak_crop.shape
            #print 'cropped_peaks.shape', cropped_peaks.shape
            cropped_peaks.iloc[peak_num, 0 : PEAK_POST_CROP_WIN + 
                            #PEAK_PRE_CROP_WIN - 1] = peak_crop.values.reshape(
                            #[len(peak_crop), ])
                            PEAK_PRE_CROP_WIN] = peak_crop.values.reshape(
                            [len(peak_crop), ])
            peak_num = peak_num + 1
            
        # copy as numpy array to use for indexing 'data1d' later
        np_peakx = peak_x.copy() 
        # convert to data-frames to concatenate into a single dataframe called 
        # peak analysis
        peak_x = pd.DataFrame({'peak_x': peak_x})
        #peak_y = pd.DataFrame({'peak_y': data1d[np_peakx]})
        peak_y = pd.DataFrame({'peak_y': data1d.iloc[np_peakx, 0]}).reset_index(drop = True)
        file_name = pd.DataFrame({'file_name': np.array([col_name]*
                                len(np_peakx))})
        selected = pd.DataFrame({'selected': np.zeros([num_peaks])})
        selected[:] = np.NaN
        #print 'peak_y.shape, selected.shape', peak_y.shape, selected.shape
        
        #cropped_peaks_all = pd.concat([cropped_peaks_all, cropped_peaks], axis = 0)
        if new_file:
            with open(results_dir+'cropped_peaks.csv', 'w') as fh:
                cropped_peaks.to_csv(fh, index = False)
            #print 'new file'
            new_file = False
        else:
            with open(results_dir+'cropped_peaks.csv', 'a') as fh:
                cropped_peaks.to_csv(fh, index = False, header = False)
            #print 'old file'
        del cropped_peaks
        
        peak_analysis_temp = pd.concat([peak_x, peak_y,
                                    file_name, selected], axis = 1)
        for feature in detect_features:
            
            feature_x = pd.DataFrame({feature + '_x': np.zeros([num_peaks])})
            feature_x[:] = np.NaN
            
            feature_rel_x = pd.DataFrame({feature + '_rel_x':
                                        np.zeros([num_peaks])})
            feature_rel_x[:] = np.NaN
            
            feature_y = pd.DataFrame({feature + '_y': np.zeros([num_peaks])})
            feature_y[:] = np.NaN
            
            peak_analysis_temp = pd.concat([peak_analysis_temp, feature_x,
                                            feature_rel_x, feature_y], axis = 1)
            
        peak_analysis = pd.concat([peak_analysis, peak_analysis_temp],
                                    axis = 0, ignore_index = True)
        col_num = col_num + 1
        
    #cropped_peaks_all.to_csv(parent_dir+'/peak_analysis_train.csv',
    #                            index = False)
                                
    #peak_analysis = peak_analysis[['peak_x', 'peak_y', 'file_name', 'selected']]
    #print 'peak_analysis.shape =', peak_analysis.shape
    #print 'peak_analysis_temp.shape =', peak_analysis_temp.shape 
    #peak_analysis = pd.concat([peak_analysis, peak_analysis_temp],
    #                                axis = 0, ignore_index = True)    
    #print 'peak_analysis.shape =', peak_analysis.shape                      
        
    return peak_analysis
    
def pt_local_extrema_w_thresh(data1d, filter_win_size, filter_sigma, 
                                peak_polarity, n_pnts_thresh, deriv2_thresh):
    # assume data is a pandas dataframe.
    
    deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
    
    if peak_polarity == 1: # look for minima in double derivative
        peak_indices = signal.argrelextrema(deriv2_filt, np.less, order = n_pnts_thresh)
        peak_indices = peak_indices[0]
        print 'Num peaks without thresh:', len(peak_indices)#, type(peak_indices)
        ###print peak_indices[0:10]
        peak_indices = [x for x in peak_indices if deriv2_filt[x] < deriv2_thresh ]
        print 'Num peaks with thresh:', len(peak_indices)#, type(peak_indices)
        print ' '
        ###print peak_indices[0:10]
    else: # look for maxima in double derivative
        peak_indices = signal.argrelextrema(deriv2_filt, np.greater, order = n_pnts_thresh)
        peak_indices = peak_indices[0]
        print 'Num peaks without thresh:', len(peak_indices)#, type(peak_indices)
        #print peak_indices[0:10]
        peak_indices = [x for x in peak_indices if deriv2_filt[x] > deriv2_thresh ]
        print 'Num peaks with thresh:', len(peak_indices)#, type(peak_indices)
        print ' '        
        #print peak_indices[0:10]
    return np.array(peak_indices)
    
def pt_filtered_deriv2(data1d, filter_win_size, filter_sigma):
    # assume data is a pandas dataframe.
    data = data1d.values # numpy array
    # reshape from (n, 1) to (n, )
    data = data.reshape([len(data), ])
    deriv1 = np.gradient(data)
    deriv2 = np.gradient(deriv1)
    deriv2_filt = pt_gaussian_filter(deriv2, filter_win_size, filter_sigma) # filtered
    # return dataframe instead of np.array   
    #deriv2_filt = pd.DataFrame(deriv2_filt, columns = ['deriv2_filt'], 
    #                           index = data1d.index) 
    return deriv2_filt

def pt_gaussian_filter(data1d, window_size, sigma):
    '''
    source: http://stackoverflow.com/questions/25571260/
    scipy-signal-find-peaks-cwt-not-finding-the-peaks-accurately
    '''
    
    window = signal.general_gaussian(window_size, sig = sigma, p = 1.0)
    ##print 'window, data1d shapes:', window.shape, data1d.shape 
    filtered = signal.fftconvolve(data1d, window)
    # filtered signal is on a much different scale: rescale after
    # convolution
   # sometimes the following can become zero.
    if np.average(data1d) != 0 and np.average(filtered) !=0:
        filtered = (np.average(data1d) / np.average(filtered)) * filtered
    # filtered data is phase shifted by 0.5* window size.
    shift_num = -1*int(np.floor(window_size/2.0))
    filtered = np.roll(filtered, shift_num)
    # filtered data is longer by window size. Keep only original length
    filtered = filtered[ : len(data1d)]
    '''    
    df_filtered = pd.DataFrame()
    for col_name in df:
        # convert to numpy array
        data1d = df[col_name].values
        # reshape from (n, 1) to (n, )
        #print col_name
        #print 'before data1d.shape', data1d.shape
        #data1d = data1d.reshape([len(data1d), ])
        data1d = data1d.reshape([len(data1d)])
        #print 'after data1d.shape', data1d.shape
        # p= 1.0 means gaussian
        window = signal.general_gaussian(window_size, sig = sigma, p = 1.0)
        filtered = signal.fftconvolve(window, data1d)
        # filtered signal is on a much different scale: rescale after
        # convolution
        filtered = (np.average(data1d) / np.average(filtered)) * filtered
        # filtered data is phase shifted by 0.5* window size.
        shift_num = -1*int(np.floor(window_size/2.0))
        filtered = np.roll(filtered, shift_num)
        # filtered data is longer by window size. Keep only original length
        filtered = filtered[ : len(data1d)]
        df_filtered_1d = pd.DataFrame({col_name:filtered})
        df_filtered = pd.concat([df_filtered, df_filtered_1d], axis = 1)
    return df_filtered
    '''
    return filtered
    
def pt_on_peak_pick(event, fig, peak_nums_rnd, num_peaks, peak_analysis, down_sample_factor, start, end, set_col_names, filter_win_size, filter_sigma,
                    PEAK_PRE_CROP_WIN, results_dir, data_folder, fn_extension, DISPLAY_WIN, yrange, 
                    detect_features, peak_num, feature_num, display_deriv2):
    ''' Stuff to do when text or trace on graph is clicked
    '''
            
    selection_str = ''
    peak_num_rnd = peak_nums_rnd[peak_num[0]]
    peak_x = peak_analysis['peak_x']
    peak_y = peak_analysis['peak_y']
    file_name = peak_analysis['file_name']
    
    if isinstance(event.artist, Text): # if clicked item is text
        text = event.artist
        selection_str = text.get_text()
        print 'peak #', peak_num[0], file_name[peak_num_rnd]
        print selection_str
        
        if selection_str == 'Accept':
            peak_analysis.loc[peak_num_rnd, ['selected']] = 1
            peak_analysis.to_csv(results_dir+'peak_vars.csv', 
                                index = False)
            
            if len(detect_features) > 0:
            # make new figure for feature selection
                fig.clear()
                sys.stdout.flush()
                fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
                the_feature = detect_features[feature_num[0]]
                pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], data1d,
                            deriv2_filt, display_deriv2, DISPLAY_WIN, yrange, pick_feature =
                            the_feature)
                sys.stdout.flush()
                return
            elif peak_num[0] + 1 < num_peaks:
                # make new figure for peak selection
                fig.clear()
                peak_num[0] = peak_num[0] + 1
                peak_num_rnd = peak_nums_rnd[peak_num[0]]
                sys.stdout.flush()
                fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
                pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], 
                            data1d, deriv2_filt, display_deriv2, DISPLAY_WIN, yrange)
                sys.stdout.flush()
                return
                
            else:
                print 'No more peaks.'
                sys.stdout.flush()
                fig.clear()
                plt.close(fig)
                return
                
                
            
        elif selection_str == 'Reject':
            peak_analysis.loc[peak_num_rnd, ['selected']] = 0
            
            # in case the feature was previously accepted, but is now rejected
            # (by going back)
            for feature in detect_features:
                
                peak_analysis.loc[peak_num_rnd, [feature + '_x']] = np.NaN
                peak_analysis.loc[peak_num_rnd, [feature + '_rel_x']] = np.NaN
                peak_analysis.loc[peak_num_rnd, [feature + '_y']] = np.NaN

                
            peak_analysis.to_csv(results_dir+'peak_vars.csv',
                                index = False)
            
            if peak_num[0] + 1 < num_peaks:
                # make new figure for peak selection
                fig.clear()
                peak_num[0] = peak_num[0] + 1
                peak_num_rnd = peak_nums_rnd[peak_num[0]]
                sys.stdout.flush()
                fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
                pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], 
                            data1d, deriv2_filt, display_deriv2, DISPLAY_WIN, yrange)
                sys.stdout.flush()
                return
                
            else:
                print 'No more peaks.'
                sys.stdout.flush()
                fig.clear()
                plt.close(fig)
                return
                
        elif selection_str == 'Back':
            if peak_num[0] - 1 >= 0:
                peak_num[0] = peak_num[0] - 1
            else:
                peak_num[0] = 0
            # go to previous peak
            fig.clear()
            sys.stdout.flush()
            peak_num_rnd = peak_nums_rnd[peak_num[0]]
            fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
            data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
            deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
            pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], data1d,
                                deriv2_filt, display_deriv2, DISPLAY_WIN, yrange)
            sys.stdout.flush()
            return
        
    elif isinstance(event.artist, Line2D): # if clicked item is data trace
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        
        
        
        ind = event.ind # has indexes of points within tolerance
        the_feature = detect_features[feature_num[0]]
        #print 'onpick points:', zip(xdata[ind], ydata[ind])
        x0 = int(round(np.median(xdata[ind])))
        y0 = ydata[x0]
        print detect_features[feature_num[0]], 'x, y =', x0, y0
        peak_analysis.loc[peak_num_rnd, [the_feature + '_x']] = x0
        peak_analysis.loc[peak_num_rnd, [the_feature + '_rel_x']] = (x0 -
                                    peak_x[peak_num_rnd] + PEAK_PRE_CROP_WIN)
        peak_analysis.loc[peak_num_rnd, [the_feature + '_y']] = y0
        peak_analysis.to_csv(results_dir+'peak_vars.csv',
                            index = False)
        
        feature_num[0] = feature_num[0] + 1
        if feature_num[0] < len(detect_features): # plot for next feature
            fig.clear()
            fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
            data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
            deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
            the_feature = detect_features[feature_num[0]]
            pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], data1d,
                        deriv2_filt, display_deriv2, DISPLAY_WIN, yrange, pick_feature = the_feature)
            sys.stdout.flush()
            return
        else:
            feature_num[0] = 0 # reset feature num
            #print 'next feature num =', feature_num[0]
            # make new figure for peak selection
            if peak_num[0] + 1 < num_peaks:
                fig.clear()
                peak_num[0] = peak_num[0] + 1
                peak_num_rnd = peak_nums_rnd[peak_num[0]]
                sys.stdout.flush()
                fn= ''.join([data_folder, file_name[peak_num_rnd], fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                deriv2_filt = pt_filtered_deriv2(data1d, filter_win_size, filter_sigma)
                pt_make_plot(peak_x[peak_num_rnd], peak_y[peak_num_rnd], data1d,
                            deriv2_filt, display_deriv2, DISPLAY_WIN, yrange)
                sys.stdout.flush()
                return  
            else:
                print 'No more peaks.'
                sys.stdout.flush()
                fig.clear()
                plt.close(fig)
                return                     
 
def pt_make_plot(peak_x0, peak_y0, data, deriv2_filt, display_deriv2, DISPLAY_WIN, yrange,
                pick_feature = None):
                
    #plt.plot(data, color = 'lightgrey')
    plt.plot()
    if display_deriv2: # show 2nd derivative on y -axis
        '''
        http://stackoverflow.com/questions/7660951/controlling-the-tracker-when-using-twinx
        Somehow the mouse click reports coordinates for whichever axis is 
        #plotted 2nd, but we want the coordinates of data and not double 
        derivative. So plotting data on 2nd axis and deriv on 1st axis.
        ''' 
        ax2= plt.gca()
        ax1 = ax2.twinx()
        # Without following, ax1 will be plotted on right axis.
        ax2.yaxis.set_ticks_position("right")
        ax1.yaxis.set_ticks_position("left")
        ax2.plot(deriv2_filt, color = 'lightgray', zorder = 1)
        

        if pick_feature != None:
            # picker = 5. Fire event if click within 5 points.
            ax1.plot(data, color = 'blue', picker = 5, zorder = 1)  
        else:
            ax1.plot(data, color = 'blue', zorder = 1)
        ax1.plot(peak_x0, peak_y0, color = 'blue', marker = 'o', linewidth = 0, zorder = 0)        
               
    else:
        ax1= plt.gca()
        if pick_feature != None:
            # picker = 5. Fire event if click within 5 points.)
            ax1.plot(data, color = 'blue', picker = 5, zorder = 1)  
        else:
            ax1.plot(data, color = 'blue', zorder = 1)
        ax1.plot(peak_x0, peak_y0, color = 'blue', marker = 'o', linewidth = 0, zorder = 0)
    #plt.plot(pd.rolling_std(data_filtered, 50, center=True), color = 'green')

    print ' '
    print 'peak_x, peak_y', peak_x0, peak_y0
    
    # restrict display range to data range
    if peak_x0 - 0.5*DISPLAY_WIN < 0:
        x1 = 0
    else:
        x1 = peak_x0 - 0.5*DISPLAY_WIN
        
    # restrict display range to data range    
    if peak_x0 + 0.5*DISPLAY_WIN > len(data):
        x2 = len(data)
    else:
        x2 = peak_x0 + 0.5*DISPLAY_WIN
    
    x1 = int(x1)
    x2 = int(x2)
    
    # set y range based on mean and stddev of data
    # mean returns a series. extract the value.      
    ##$##filt_mean = data_filtered[x1 : x2 +1].mean(axis = 0)[0]
    ##$##filt_std = data_filtered[x1 : x2 + 1].std(axis = 0)[0]
    data_mean = data[x1 : x2 +1].mean(axis = 0)[0]
    data_std = data[x1 : x2 + 1].std(axis = 0)[0]
    if yrange == None:
        ax1.set_ylim(data_mean - 10*data_std, data_mean + 10*data_std)
    else:
        ax1.set_ylim(data_mean + yrange[0], data_mean + yrange[1])
        
    #deriv2_mean = deriv2_filt[x1 : x2 +1].mean(axis = 0)[0]
    #deriv2_std = deriv2_filt[x1 : x2 + 1].std(axis = 0)[0]
    #ax2.set_ylim(deriv2_mean - 10*deriv2_std, deriv2_mean + 10*deriv2_std)
    
    #ax = plt.gca()
    #plt.xlim(peak_x0 - 0.5*DISPLAY_WIN, peak_x0 + 0.5*DISPLAY_WIN)
    plt.xlim(x1, x2)
    
    if pick_feature:
        plt.text(0.6, 0.9, 'Choose ' + pick_feature, fontsize = 15, 
                transform=ax1.transAxes)
    else:
    #http://matplotlib.org/1.4.2/examples/event_handling/pick_event_demo.html
    # coordinates in terms of axis range from  0 to 1.
        plt.text(0.7, 0.9, 'Accept', fontsize = 15, 
                transform=ax1.transAxes, picker=True)
        plt.text(0.85, 0.9, 'Reject', fontsize = 15, 
                transform=ax1.transAxes, picker=True)
        plt.text(0.7, 0.8, 'Back', fontsize = 15, 
                transform=ax1.transAxes, picker=True)
    plt.draw()
    sys.stdout.flush()



def pt_display_on_peak_pick(event, fig, num_peaks, num_files, peak_analysis, down_sample_factor, start, end, set_col_names, filter_win_size, filter_sigma,
                    data_folder, fn_extension, DISPLAY_WIN, yrange, 
                    detect_features, peak_num, file_num, plot_full_traces = False):
    ''' Stuff to do when text or trace on graph is clicked
    '''
            
    selection_str = ''
    selected = peak_analysis['selected']
    file_name = peak_analysis['file_name']
    file_names = file_name.unique()
    peak_x = peak_analysis['peak_x']
    peak_y = peak_analysis['peak_y']
    
     
    if plot_full_traces:
        if isinstance(event.artist, Text): # if clicked item is text
            text = event.artist
            selection_str = text.get_text()
            print 'file #', file_num[0], file_names[file_num[0]]
            #print selection_str
            sys.stdout.flush()
            
            if selection_str == 'Next':
                if file_num[0] + 1 < num_files:
                    file_num[0] = file_num[0] + 1
                else:
                    print 'No more files.'
                # make new figure for feature selection
                fig.clear()
                sys.stdout.flush()
                the_file_name = file_names[file_num[0]]
                fn= ''.join([data_folder, the_file_name, fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                #data1d_filtered = pt_gaussian_filter(data1d, filter_win_size, filter_sigma)
                
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
                    
                #print peak_x[peak_num[0]], peak_y[peak_num[0]]  
                #print features_xy
                sys.stdout.flush()         
                pt_display_make_plot(peak_x0, peak_y0, features_xy, selected0, the_file_name, data1d, 
                            DISPLAY_WIN, yrange, 
                            plot_full_traces = plot_full_traces)
                            
                sys.stdout.flush()
                return
                
            elif selection_str == 'Back':
                
                if file_num[0] - 1 >= 0:
                    file_num[0] = file_num[0] - 1
                else:
                    print 'No more files.'
                # make new figure for feature selection
                fig.clear()
                sys.stdout.flush()
                the_file_name = file_names[file_num[0]]
                fn= ''.join([data_folder, the_file_name, fn_extension])
                data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
                #data1d_filtered = pt_gaussian_filter(data1d, filter_win_size, filter_sigma)
                
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
                    
                sys.stdout.flush()                 
                pt_display_make_plot(peak_x0, peak_y0, features_xy, selected0, the_file_name, data1d, 
                            DISPLAY_WIN, yrange, 
                            plot_full_traces = plot_full_traces)
                            
                sys.stdout.flush()
                return
                    
            else:
                pass
##==========================
        
    if isinstance(event.artist, Text): # if clicked item is text
        text = event.artist
        selection_str = text.get_text()
        print 'peak #', peak_num[0], file_name[peak_num[0]]
        #print selection_str
        sys.stdout.flush()
        
        if selection_str == 'Next':
            if peak_num[0] + 1 < num_peaks:
                peak_num[0] = peak_num[0] + 1
            else:
                print 'No more peaks.'
            # make new figure for feature selection
            fig.clear()
            sys.stdout.flush()
            fn= ''.join([data_folder, file_name[peak_num[0]], fn_extension])
            data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
            #data1d_filtered = pt_gaussian_filter(data1d, filter_win_size, filter_sigma)
            
            features_xy = []
            for the_feature in detect_features:
                
                x0 = peak_analysis.loc[peak_num[0], [the_feature + '_x']]
                y0 = peak_analysis.loc[peak_num[0], [the_feature + '_y']]
                features_xy.append((x0, y0))
                
            #print peak_x[peak_num[0]], peak_y[peak_num[0]]  
            #print features_xy
            sys.stdout.flush()         
            pt_display_make_plot(peak_x[peak_num[0]], peak_y[peak_num[0]], features_xy, selected[peak_num[0]], file_name[peak_num[0]], data1d, 
                        DISPLAY_WIN, yrange, 
                        plot_full_traces = plot_full_traces)
                        
            sys.stdout.flush()
            return
            
        elif selection_str == 'Back':
            
            if peak_num[0] - 1 >= 0:
                peak_num[0] = peak_num[0] - 1
            else:
                print 'No more peaks.'
            # make new figure for feature selection
            fig.clear()
            sys.stdout.flush()
            fn= ''.join([data_folder, file_name[peak_num[0]], fn_extension])
            data1d = pt_load_data(fn, down_sample_factor, start, end, set_col_names = set_col_names) # raw
            #data1d_filtered = pt_gaussian_filter(data1d, filter_win_size, filter_sigma)
            
            features_xy = []
            for the_feature in detect_features:
                
                x0 = peak_analysis.loc[peak_num[0], [the_feature + '_x']]
                y0 = peak_analysis.loc[peak_num[0], [the_feature + '_y']]
                features_xy.append((x0, y0))
                            
            pt_display_make_plot(peak_x[peak_num[0]], peak_y[peak_num[0]], 
                                features_xy, selected[peak_num[0]], 
                                file_name[peak_num[0]], data1d,
                                DISPLAY_WIN, yrange, 
                                plot_full_traces = plot_full_traces)
                        
            sys.stdout.flush()
            return
                
        else:
            pass

def pt_display_make_plot(peak_x0, peak_y0, features_xy, selected0, file_name0, 
                        data, DISPLAY_WIN, yrange, 
                        plot_full_traces = False):
    
    #print 'plot_full_traces', plot_full_traces            
    plt.plot(data, color = 'red')
    #plt.plot(data_filtered, color = 'red')
    
    print ' '
    if plot_full_traces == False:
        print 'peak_x, peak_y', peak_x0, peak_y0
    
    
    
    #if plotting full trace
    if plot_full_traces:
        x1 = 0
        x2 = len(data) -1
        #print 'setting x0, x1 values', x1, x2
    else:
        # restrict display range to data range
        if peak_x0 - 0.5*DISPLAY_WIN < 0:
            x1 = 0
        else:
            x1 = peak_x0 - 0.5*DISPLAY_WIN
            
        # restrict display range to data range    
        if peak_x0 + 0.5*DISPLAY_WIN > len(data):
            x2 = len(data)
        else:
            x2 = peak_x0 + 0.5*DISPLAY_WIN
        
        x1 = int(x1)
        x2 = int(x2)        
    
    # set y range based on mean and stddev of data
    # mean returns a series. extract the value.      
    filt_mean = data[x1 : x2 +1].mean(axis = 0)[0]
    filt_std = data[x1 : x2 + 1].std(axis = 0)[0]
    if yrange == None:
        plt.ylim(filt_mean - 10*filt_std, filt_mean + 10*filt_std)
    else:
        plt.ylim(filt_mean + yrange[0], filt_mean + yrange[1])
    
    
    if plot_full_traces == False:
        if selected0 == 0:
            plt.plot(peak_x0, peak_y0, color = 'red', marker = 'o', linewidth = 0, markersize = 10)
            for (feature_x, feature_y) in features_xy:
                # convert from pandas series to value. Useful if value = NaN
                feature_x = feature_x[0]
                feature_y = feature_y[0]
                if np.isnan(feature_x) or np.isnan(feature_y):
                    print ('Warning!', 'feature lies outside cropped peaks.'  
                        'Increase range of cropped peaks to include all features')
                plt.plot(feature_x, feature_y, color = 'red', marker = '*', linewidth = 0, markersize = 10)
                
        else:
            plt.plot(peak_x0, peak_y0, color = 'green', marker = 'o', linewidth = 0, markersize = 10)
            for (feature_x, feature_y) in features_xy:
                # convert from pandas series to value. Useful if value = NaN
                feature_x = feature_x[0] 
                feature_y = feature_y[0]
                if np.isnan(feature_x) or np.isnan(feature_y):
                    print ('Warning!', 'feature lies outside cropped peaks.'  
                        'Increase range of cropped peaks to include all features')
                plt.plot(feature_x, feature_y, color = 'green', marker = '*', linewidth = 0, markersize = 10)
    else:
        colors = ['green' if the_selection else 'red' for the_selection in selected0]
        # plt.plot uses same color for all points
        plt.scatter(peak_x0, peak_y0, color = colors, marker = 'o', linewidth = 0, s = 40)
        for (feature_x, feature_y) in features_xy:
            # convert from pandas series to value. Useful if value = NaN
            #print 'feature_x, feature_y', type(feature_x), type(feature_y)
            #print feature_x
            #print feature_y
            #feature_x = feature_x[0]
            #feature_y = feature_y[0]
            
            #if np.isnan(feature_x) or np.isnan(feature_y):
            #    print ('Warning!', 'feature lies outside cropped peaks.'  
            #            'Increase range of cropped peaks to include all features')
            plt.scatter(feature_x.values, feature_y.values, color = colors, marker = '*', s = 40)
    
    
    ax = plt.gca()
    if plot_full_traces == True:
        plt.xlim(x1, x2)
    else:        
        plt.xlim(peak_x0 - 0.5*DISPLAY_WIN, peak_x0 + 0.5*DISPLAY_WIN)
    

    #http://matplotlib.org/1.4.2/examples/event_handling/pick_event_demo.html
    # coordinates in terms of axis range from  0 to 1.
    
    plt.text(0.7, 0.9, 'Back', fontsize = 15, transform=ax.transAxes,
            picker=True)
    plt.text(0.85, 0.9, 'Next', fontsize = 15, transform=ax.transAxes,
            picker=True)
    plt.text(0.6, 0.8, file_name0, fontsize = 15, transform=ax.transAxes)        
            
            
    plt.draw()
    sys.stdout.flush()
