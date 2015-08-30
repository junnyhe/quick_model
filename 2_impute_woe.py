import csv
import gzip
import os
import sys
import datetime
import random
from numpy import *
sys.path.append("tools/csv_operations")
import csv_ops
from csv_ops import *
from csv_woe_cat import *
from csv_impute import *
from multiprocessing import Pool


global work_dir


work_dir='/Users/junhe/Dropbox/Python_Work/modeling/data/' # everything should/will be in w
support_dir='/Users/junhe/Dropbox/Python_Work/modeling/code/support_files/'

################################################################################
# Perform imputation                                                           #
# 1. Calculate statistics, create missing value mapping                        #
################################################################################

ins_file="data1_train.csv.gz" # in sample data used to calculate the stats
imp_median_var_list_file=support_dir+"var_list_impute_median.csv" # var list for imputing missing to median 
imp_zero_var_list_file=support_dir+"var_list_impute_zero.csv" # var list for imputeing missing to zero

impute_create_mapping(work_dir,ins_file,imp_median_var_list_file,imp_zero_var_list_file)

input_file="data1_train.csv.gz"
output_file="data1_train_imp.csv.gz"
impute_replace_pickle(work_dir,input_file,output_file)

input_file="data1_val.csv.gz"
output_file="data1_val_imp.csv.gz"
impute_replace_pickle(work_dir,input_file,output_file)

input_file="data2.csv.gz"
output_file="data2_imp.csv.gz"
impute_replace_pickle(work_dir,input_file,output_file)


#csv_EDD(work_dir+'model_data_pmt_ins_ds_imp.csv.gz')

################################################################################
# Create WOE variables                                                         #
# 1. Calculate statistics, create WOE mapping                                  #
################################################################################

input_file='data1_train_imp.csv.gz'
woe_var_list_file=support_dir+'var_list_woe.csv'

risk_table(work_dir, input_file, woe_var_list_file, target='target', smooth_num=100, target_value='1')

################################################################################
# Create WOE variables                                                         #
# 2. Create WOE variables                                                      #
################################################################################

input_file="data1_train_imp.csv.gz"
output_file="data1_train_imp_woe.csv.gz"
woe_assign_pickle(work_dir,input_file,output_file)

input_file="data1_val_imp.csv.gz"
output_file="data1_val_imp_woe.csv.gz"
woe_assign_pickle(work_dir,input_file,output_file)

input_file="data2_imp.csv.gz"
output_file="data2_imp_woe.csv.gz"
woe_assign_pickle(work_dir,input_file,output_file)


################################################################################
# check data quality, determine modeling var list                              #
################################################################################

csv_EDD(work_dir+'data1_train_imp_woe.csv.gz')


