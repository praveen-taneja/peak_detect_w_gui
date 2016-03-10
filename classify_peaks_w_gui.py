import pandas as pd
import matplotlib.pylab as plt
import sklearn.cross_validation as cross_validation
import sklearn.preprocessing as preprocessing
import sys
import sklearn.metrics as metrics

def classify_peaks(df, results_dir, clf, label_column, PEAK_PRE_CROP_WIN, 
                   PEAK_POST_CROP_WIN, FEATURE_START_COLNUM, 
                   FEATURE_END_COLNUM):

  
    df_train = df[pd.notnull(df[label_column])] # with label
    df_predict = df[pd.isnull(df[label_column])] # no label
    train_nrows = df_train.shape[0]
    
    
    # label frequency
    grouped = df_train.groupby([label_column])    
    print '  '
    print 'label frequency:'
    for name, group in grouped:
        percent_freq = 100*group[label_column].size/float(train_nrows)
        print name, '----', "{0:0.2f}".format(percent_freq), '%'
        
    # plot selected and rejected peaks
    if label_column == 'selected':
        plt.plot(df_train[df[label_column] == 1].iloc
        [:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].transpose(), color = 'green')
        plt.plot(df_train[df[label_column] == 0].iloc
        [:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].transpose(), color = 'red')
        plt.show()
    
    labels = df_train[label_column].values
    features = df_train.iloc[:, FEATURE_START_COLNUM:FEATURE_END_COLNUM].values
    
    # standardize features
    scaler = preprocessing.StandardScaler().fit(features)
    #print scaler.mean_ 
    #print scaler.std_ 
    features = scaler.transform(features)
    
    x_train, x_train_test, y_train, y_train_test = (cross_validation.
                                            train_test_split(features, labels, 
                                            test_size=0.1))
                                            
   
    cv_score = cross_validation.cross_val_score(clf, x_train, y_train, cv = 5, 
                                                verbose = 0)
    print 'cross validation accuracy =', cv_score
    print 'mean cross validation accuracy =', cv_score.mean()
    sys.stdout.flush()
    
    clf.fit(x_train, y_train)
    
    # predict labels for held out test data
    #y_train_test_predict = clf.predict(x_train_test)
                                                
    print 'accuracy (held out data) =', clf.score(x_train_test, y_train_test) 
    #metrics.accuracy_score(y_train_test, y_train_test_predict, normalize = True)
    sys.stdout.flush()
    print ' '
    
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
