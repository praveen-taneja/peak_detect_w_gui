General idea: 1st we generate training data. Then we classify and predict. Finally, we visualize predictions. 

1. Start by running the file peak_detect_dialog_w_gui.


Training data

Some important parameters to adjust are listed below. Some trial and error is need for for each dataset. Parameters can be saved and loaded for future use.


a) Filter (gaussian) – Window size and width. Adjust so that the noise is removed but signal is not affected much. Window size should be about 3 times the window width. Note: It is applied to the double derivative and not the original signal.

b) Thresholds (for double derivative)– Preliminary peak detection is using double derivative. 2nd derivative has a minima when the original signal has a maxima (and vice versa). To visualize both the original signal and 2nd derivative, check the check box in the 'training data' panel. Adjust so that we have false positives but no false negatives at this stage. Smaller values of 1st threshold will detect more peaks in the 2nd derivative and consequently more peaks in the signal. 

c) Display range - Adjusting the display range using properly is crucial because this affects our perception of what is or isn’t a peak quite drastically. The y-range is relative to mean value of signal in x-range.  If X-range is longer than full length of signal, full signal is displayed. This can be used if we need to view peaks in relation to full signal.

d) Range – To select s sub-range of signal in which to look for peaks.

e) Crop – Number of points to crop before and after the peak for training. Must be enough to include features in the feature list (like peak threshold) for each peak.

Tips:

1. It is important to be as consistent as possible when generating training data.

2. Cross validation accuracy assumes that the training data is correct. But it's possible that multiple labeling of the same training data leads to different labels. The cross-validation should be compared against this human accuracy.