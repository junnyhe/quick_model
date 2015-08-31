import pandas as pd
import numpy as np
from numpy import *
import sys
sys.path.append("tools/model_tools")
sys.path.append("tools/csv_operations")

from sklearn.ensemble import RandomForestClassifier
from model_performance_evaluation import performance_eval_train_validation
from model_performance_evaluation import performance_eval_test_downsample

var_list=['signal_36','signal_140','signal_408','signal_158','signal_400','signal_371','signal_248','signal_59','signal_38','signal_410','signal_418','signal_420','signal_428','signal_159','signal_409','signal_609','signal_419','signal_143','amount','signal_429','signal_414','signal_411','signal_401','signal_404','signal_415','signal_424','signal_421','signal_37','signal_405','signal_425','signal_416','signal_412','signal_561','signal_417','signal_560','signal_413','signal_402','signal_406','signal_422','signal_426','signal_407','signal_403','signal_427','signal_423','signal_128','signal_33','signal_175','signal_157','signal_597','signal_228','signal_596','signal_593','signal_174','signal_592','signal_600','signal_27','signal_633','signal_595','signal_624','signal_505','signal_524','signal_127','signal_591','signal_623','signal_306','signal_594','signal_590','signal_247','signal_307','signal_628','signal_532','signal_304','signal_627','signal_182','signal_303','signal_154','signal_144','signal_181','signal_302','signal_305','signal_24','signal_533','signal_534','signal_142','signal_531','signal_535','signal_504','signal_523','signal_538','signal_632','signal_703','signal_536','signal_620','signal_631','signal_702','signal_700','signal_204','signal_155','signal_141','signal_507','signal_508','signal_711','signal_708','signal_710','signal_149','signal_173','signal_176','signal_537','signal_701','signal_709','signal_715','signal_525','signal_527','signal_706','signal_707','signal_714','signal_512','signal_513','signal_704','signal_705','signal_712','signal_713','signal_50','signal_153','signal_152','signal_156','signal_509','signal_510','signal_511','signal_526','signal_528','signal_529','signal_625','signal_629','signal_28','signal_34','signal_49','signal_150','signal_514','signal_626','signal_630','signal_636','signal_148','signal_1','signal_4','signal_11','signal_12','signal_17','signal_18','signal_19','signal_26','signal_29','signal_31','signal_35','signal_361','signal_362','signal_607','lo_signal_48','lo_signal_179','lo_signal_300','lo_signal_506','lo_signal_547','lo_signal_622','lo_signal_610','lo_signal_180','lo_signal_301','lo_signal_47','lo_signal_638','lo_signal_355','lo_signal_2','lo_signal_580','lo_signal_621','lo_signal_4']

output_dir="../test/"

################### training #####################
# load train data
df = pd.read_csv('../data/data1_val_imp_woe.csv.gz',compression="gzip")
field_names=df.columns.values
field_types=df.dtypes
X=df[var_list]
y=df['target']

# fit
model=RandomForestClassifier(max_depth=None, n_estimators=200, max_features="auto",random_state=0,n_jobs=-1)
model.fit(X,y)

# pred
y_pred = model.predict(X)
p_pred = model.predict_proba(X)
p_pred = p_pred[:,1]

# performance train
output_suffix="train"
ks, auc, lorenz_curve_capt_rate = performance_eval_test_downsample(y,p_pred,output_dir,output_suffix,good_downsample_rate=1)


###################### test ######################
# load test data
dft = pd.read_csv('../data/data2_imp_woe.csv.gz',compression="gzip")
Xt=dft[var_list]
yt=dft['target']

# pred test
yt_pred = model.predict(Xt)
pt_pred = model.predict_proba(Xt)
pt_pred = pt_pred[:,1]

# performance test
output_suffix="test"
ks, auc, lorenz_curve_capt_rate = performance_eval_test_downsample(yt,pt_pred,output_dir,output_suffix,good_downsample_rate=1)

