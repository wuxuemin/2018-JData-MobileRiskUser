import pandas as pd
import numpy as np
import xgboost as xgb  
from sklearn import metrics
from pandas import Series, DataFrame
from numpy import array
from sklearn.metrics import f1_score 
train = pd.read_csv('/home/wxm/Downloads/data_mining/feature_engineer/train_feature.csv')
test = pd.read_csv('/home/wxm/Downloads/data_mining/feature_engineer/test_feature.csv')
train.drop(train.columns[0], axis=1,inplace=True) 
test.drop(test.columns[0], axis=1,inplace=True) 
y_train=train.label
x_train2 = train.drop(['uid','label'],axis=1)
x_test2 = test.drop(['uid'],axis=1)


xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'eta':0.11,
    'max_depth':4,
    'min_child_weight':1,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'reg_alpha':0.001,
    'scale_pos_weight':5,
    'stratified':True,
     'gamma':0,
     #'lambda':1,
    'seed':27,
    'silent':1
}
def evalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
    res = 0.6*auc +0.4*f1
    return 'res',res

dtrain = xgb.DMatrix(x_train2,label=y_train)
xgb.cv(xgb_params,dtrain,num_boost_round=190,nfold=3,verbose_eval=10,early_stopping_rounds=100,maximize=True,feval=evalMetric)


model=xgb.train(xgb_params,dtrain=dtrain,num_boost_round=190,verbose_eval=10,evals=[(dtrain,'train')],maximize=True,feval=evalMetric,early_stopping_rounds=100)
dtest = xgb.DMatrix(x_test2)
preds =model.predict(dtest)
df =pd.DataFrame({'uid':test.uid,'label':preds})
df=df.sort_values(by='label',ascending=False)
df.label=df.label.map(lambda x: 1 if x>=0.5 else 0)
df.label = df.label.map(lambda x: int(x))

df.to_csv('/home/wxm/Downloads/data_mining/result/submission.csv',index=False,header=False,sep=',',columns=['uid','label'])


