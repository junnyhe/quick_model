import csv
import gzip
import sys
import numpy as np
import time
import pickle
from numpy import *
#import matplotlib.pyplot as pl
import random
from sklearn import tree
from operator import itemgetter
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.metrics import roc_curve, auc
sys.path.append("tools/model_tools")
sys.path.append("tools/csv_operations")

import csv_ops
from csv_ops import *
from load_data import *
#from getAUC import *
#from ks_roc import *
from model_performance_evaluation import performance_eval_train_validation
from model_performance_evaluation import performance_eval_test
from model_performance_evaluation import performance_eval_test_downsample


def model_train_validation(ins_file, oos_file, classifier, var_list_filename, output_dir, outpu):
    """
    train model
    evaluate on the train and validation data
    evaluate the model performance on the train and validation data
    """
    #################### Load train and validation data ####################
    print 'Loading data for modeling starts ...'
    t0=time.time()
    target_name='target'
    X,y = load_data_fast(ins_file, var_list_filename, target_name)
    Xv,yv = load_data_fast(oos_file, var_list_filename, target_name)
    print "Loading data done, taking ",time.time()-t0,"secs"
    
    # prepare trivial input values for generating reason code in production
    trivial_input_values_file = output_dir+'trivial_input_values.p'
    trivial_input_values = median(X,axis=0)
    pickle.dump(trivial_input_values,open(trivial_input_values_file,'wb'))
    
    # Train Model
    print '\nModel training starts...'
    t0=time.time()
    model = classifier
    model.fit(X, y)
    print "Model training done, taking ",time.time()-t0,"secs"
    pickle.dump(model,open(output_dir+"model.p",'wb')) # save model to disk
    
    '''
    #export to tree graph in DOT format, tree only
    tree.export_graphviz(model,out_file=output_dir+'tree.dot')
    os.system("dot -Tpng "+output_dir+"tree.dot -o "+output_dir+"tree.png")
    '''
    
    # Predict Train
    y_pred = model.predict(X)
    p_pred = model.predict_proba(X)
    p_pred = p_pred[:,1]
    
    # Predict Validation
    yv_pred = model.predict(Xv)
    pv_pred = model.predict_proba(Xv)
    pv_pred = pv_pred[:,1]
    
    # Performance Evaluation: Train and Validation
    ks, auc, lorenz_curve_capt_rate = performance_eval_train_validation(y,p_pred,yv,pv_pred,output_dir,output_suffix)
    
    
    #################### Random Forest Feature Importance ######################
    try:
        varlist_file=open(var_list_filename,'rU')
        varlist_csv=csv.reader(varlist_file)
        var_list=[]
        for row in varlist_csv:
            var_list.append(row[0])
        out_feat_import = open(output_dir + 'feature_import_' + str(output_suffix)+'.csv', 'wb')
        feat_import_csv = csv.writer(out_feat_import)
        var_import = zip(range(len(var_list)),var_list,model.feature_importances_)
        feat_import_csv.writerow(['var seq num','var name','importance'])
        print "RandomForest classifier, var importance was output"
        for row in var_import:
            feat_import_csv.writerow(row)
    except:
        print "Not RandomForest classifier, var importance not created"
    
    
    return ks, auc, lorenz_curve_capt_rate

    

def model_test_data_evaluation_comp_ruletag(test_data_file, var_list_filename, model_file, output_dir, output_suffix, good_downsample_rate):
    
    #################### Load Model and Evaluate Performance ##################
    ############################### Test Data #################################
    # Ad Hoc code
    # compare model results with rules
    
    # Load Test Data
    print 'Loading test data starts ...'
    t0=time.time()
    target_name='target'
    key_name='payment_request_id'
    tag_name='manual_review'
    X,y,key,tag = load_data_with_key_tag_fast(test_data_file, var_list_filename, target_name, key_name, tag_name)
    print "Loading test data done, taking ",time.time()-t0,"secs"
    
    # Load Model
    print 'Loading model ...'
    t0=time.time()
    model = pickle.load(open(model_file,'rb'))
    
    # Predict Test Data
    y_pred = model.predict(X)
    p_pred = model.predict_proba(X)
    p_pred = p_pred[:,1]

    # Performance Evaluation: Test
    print 'Evalutate model performance ...'
    ks, auc, lorenz_curve_capt_rate = performance_eval_test_downsample(y,p_pred,output_dir,output_suffix,good_downsample_rate)
    
    
    return ks, auc, lorenz_curve_capt_rate
    

    
    

data_dir='/Users/junhe/Dropbox/Python_Work/modeling/data/'
support_dir='/Users/junhe/Dropbox/Python_Work/modeling/code/support_files/'
result_dir='/Users/junhe/Dropbox/Python_Work/modeling/results/'

good_downsample_rate = 0.05 #used to scale back hit rate


########################### Instantiate Classifiers ############################


classifiers = {
    "Logistic":LogisticRegression(),
    "NearestNeighbors":KNeighborsClassifier(100),
    "LinearSVM":SVC(kernel="linear", C=0.025),
    "RBFSVM":SVC(gamma=2, C=1),
    "DecisionTree":DecisionTreeClassifier(max_depth=32),
    "RandomForest":RandomForestClassifier(max_depth=None, n_estimators=200, max_features="auto",random_state=0,n_jobs=-1),
    "RandomForest2":RandomForestClassifier(max_depth=8, n_estimators=200, max_features="auto",random_state=0,n_jobs=-1),
    "RandomForest3":RandomForestClassifier(min_samples_leaf=4, n_estimators=400, max_features="auto",random_state=0,n_jobs=-1),
    "AdaBoost":AdaBoostClassifier(n_estimators=500,random_state=0),
    "GradientBoost":GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=None, random_state=0),
    "NaiveBayes":GaussianNB(),
    "LDA":LDA(),
    "QDA":QDA()
    }

joblist=[
        (classifiers["RandomForest3"],'RF4','model_var_list_signal.csv'), # suffix and varlist
        ]

    
############################# Main: Run Different Classifiers ################################

for job in joblist:
    
    # Train Model and Evaluate Performance on Train and Validation Data
    classifier=job[0]
    output_suffix=job[1]
    var_list_filename=support_dir+job[2]
    
    
    output_dir=result_dir+output_suffix+"/"
    if os.path.exists(output_dir):
        print "results folder:",output_dir," already exist"
    else:
        print "results folder:",output_dir," not exist; will be created"
        os.system("mkdir "+output_dir.replace(' ','\ '))

    ins_file=data_dir+'data1_train_imp_woe.csv.gz'
    oos_file=data_dir+'data1_val_imp_woe.csv.gz'
    
    
    ks, auc, lorenz_curve_capt_rate = model_train_validation(ins_file, oos_file, classifier, var_list_filename, output_dir, output_suffix)


    # Load Model and Evaluate Performance on Test Data
    test_data_file = data_dir+'data1_train_imp_woe.csv.gz'
    model_file = output_dir+"model.p"
    output_suffix = job[1]+'_train'
    ks, auc, lorenz_curve_capt_rate= model_test_data_evaluation_comp_ruletag(test_data_file, var_list_filename, model_file, output_dir, output_suffix,good_downsample_rate)

    # Load Model and Evaluate Performance on Test Data
    test_data_file = data_dir+'data1_val_imp_woe.csv.gz'
    model_file = output_dir+"model.p"
    output_suffix = job[1]+'_val'
    ks, auc, lorenz_curve_capt_rate = model_test_data_evaluation_comp_ruletag(test_data_file, var_list_filename, model_file, output_dir, output_suffix,good_downsample_rate)

    # Load Model and Evaluate Performance on Test Data
    test_data_file = data_dir+'data2_imp_woe.csv.gz'
    model_file = output_dir+"model.p"
    output_suffix = job[1]+'_data2'
    ks, auc, lorenz_curve_capt_rate = model_test_data_evaluation_comp_ruletag(test_data_file, var_list_filename, model_file, output_dir, output_suffix,good_downsample_rate)





