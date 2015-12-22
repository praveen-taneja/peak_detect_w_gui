from sklearn import svm
from sklearn.cross_validation import cross_val_score
import pandas as pd
import matplotlib.pylab as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import os
import sys
import numpy as np
#from sklearn.metrics import accuracy_score
import collections
import peak_detect_utils1 as pd_utils
import imp
imp.reload(pd_utils)



#######Start - Variables to be edited by user #####

# Path for input data file without file suffix


#Use "/" for path separator (not back-slash)
#data_file = ('H:/Core Labs/Electrophysiology/PraveenT/MuckeLab/Dravet/spont_and_mini_epsc_ipsc/text_data/sepsc_analysis/sepsc_train_all.csv')

#====================
# for peak prediction
#====================
'''
label_column = 'selected'
# use ExtraTreesClassifier for class prediction (eg. peak selected or not)
clf = ExtraTreesClassifier(n_estimators=100, max_depth=None, 
        min_samples_split=1, random_state=0)
# number of points cropped before and after peak for training classifier

PEAK_PRE_CROP_WIN = 80
PEAK_POST_CROP_WIN = 50
FEATURE_START_COLNUM = 0


#====================
# for threshold prediction
#====================

label_column = 'threshold_rel_x'
# use svm.SVR for real value prediction (eg. peak threshold)
clf = svm.SVR(gamma=0.001, C=100.)
# number of points cropped before and after peak for training classifier
PEAK_PRE_CROP_WIN = 80
PEAK_POST_CROP_WIN = 50
FEATURE_START_COLNUM = 0


FEATURE_END_COLNUM = FEATURE_START_COLNUM + PEAK_PRE_CROP_WIN + PEAK_POST_CROP_WIN
'''
#accuracy_score = clf.score


#clf = linear_model.LogisticRegression(C=1.)
#clf = LogisticRegression(C=100000.0, class_weight=None, dual=False,
#          fit_intercept=True, intercept_scaling=1, penalty='l2',
#          random_state=None, tol=0.0001)
#detect_features = ['threshold'] # add more to the list if needed.

#######End - Variables to be edited by user #######
###################################################

def classify_peaks(df, results_dir, clf, label_column, PEAK_PRE_CROP_WIN, PEAK_POST_CROP_WIN, 
                    FEATURE_START_COLNUM, FEATURE_END_COLNUM):
    
    #print type(clf), clf
    ##data_file = pd_utils.get_path('*.*', 'Choose training data file')
    ##print 'loading data file', data_file
    #parent_dir = os.path.abspath(os.path.join(data_file, os.pardir))
    #parent_dir = parent_dir + '/'
    
    ##data_folder = os.path.split(data_file)[0]
    ##data_folder = data_folder + '/'
    
    ##classific_pars = collections.OrderedDict()
    
    ##classific_pars['data_file'] = data_file
    ##classific_pars['PEAK_PRE_CROP_WIN'] = PEAK_PRE_CROP_WIN
    ##classific_pars['PEAK_POST_CROP_WIN'] = PEAK_POST_CROP_WIN
    ##classific_pars['FEATURE_START_COLNUM'] = FEATURE_START_COLNUM
    ##classific_pars['FEATURE_END_COLNUM'] = FEATURE_END_COLNUM
    ##classific_pars['label_column'] = label_column
    ##classific_pars['clf'] = clf
    
    # save classific parameters to disk for future reference:
    ##with open(data_folder+label_column+'_classific_pars.txt', 'w') as fh:
    ##    fh .writelines('{}:{}\n'.format(k,v) for k, v in classific_pars.items())
    
    
    ##df = pd.read_table(train_fn, delimiter = ',')
    
    df_train = df[pd.notnull(df[label_column])] # with label
    df_predict = df[pd.isnull(df[label_column])] # no label
    
    len_manually_selected = len(df_train)
    #print 'len_manually_selected', len_manually_selected
    #print df_train.shape
    #print df_predict.shape
    
    # http://www.markhneedham.com/blog/2013/11/09/python-making-scikit-learn-and-
    # pandas-play-nice/
    grouped = df_train.groupby([label_column])
    
    print '  '
    print 'label frequency:'
    for name, group in grouped:
        percent_freq = 100*group[label_column].size/float(len_manually_selected)
        print name, '----', "{0:0.2f}".format(percent_freq), '%'
    
    
    labels = df_train[label_column].values
    #print labels[0:5]
    
    features = df_train.iloc[:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].values
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, 
                                            test_size=0.1)#, random_state=42)
                                    
    ##print 'x_train.shape =', x_train.shape
    ##print 'x_test.shape =', x_test.shape
    ##print 'y_train.shape =', y_train.shape
    ##print 'y_test.shape =', y_test.shape
    
    # calculate mean and std based on training data set and apply to training and
    # testing data set
    scaler = preprocessing.StandardScaler().fit(x_train)
    #print scaler.mean_ 
    #print scaler.std_ 
    
    plt.plot(np.transpose(x_train))
    plt.show()
    
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    clf.fit(x_train, y_train)
    
    # predict labels for test data
    y_predict = clf.predict(x_test)
    
    print ' '
    # normalize option is needed when using sklearn.metrics.accuracy_score
    #accur = accuracy_score(y_test, y_predict, normalize = True)
    #accur = accuracy_score(y_test, y_predict)
    #print 'accuracy =', accur
    
    
    '''
    print 'y_test, y_predict'
    for y0, y1 in zip(y_test, y_predict):
        print y0, y1
    '''
    
    # IN THE FOLLOWING FEATURES NEED TO BE STANDARDIZED
    clf_score = cross_val_score(clf, x_train, y_train, cv = 5, verbose = 0)#.mean()
    print 'cross validation accuracy =', clf_score
    print 'mean cross validation accuracy =', clf_score.mean()
    sys.stdout.flush()
    
    # now predict labels for data with no labels
    labels = df_predict[label_column].values
    features = df_predict.iloc[:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].values
    
    x_predict = scaler.transform(features)
    y_predict = clf.predict(x_predict)
    
    # add predictions to data frame
    df_predict.loc[:, label_column] = y_predict
    df = pd.concat([df_train, df_predict])
    df = df.sort(columns = ['file_name', 'peak_x'])
    df.to_csv(results_dir + 'predicted.csv', index = False)
    
    '''
    features = df.iloc[:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].values
    labels = df[label_column].values
    print features.shape, labels.shape
    fig = plt.figure()
    plt.plot(np.transpose(features), c = plt.get_cmap('jet')(labels))
    plt.show()
    '''