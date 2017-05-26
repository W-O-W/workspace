from data_loader import *
import numpy as np
path="/home/ki/Desktop/pre"
data_train=data_loader(path+"/data_train.csv","CSV")
data_test=data_loader(path+"/data_test.csv","CSV")
"""
#确定appID并添加appCategory
data_ad=data_loader(path+"/ad.csv","CSV")
data_ad.index=data_ad.creativeID

data_ca=data_loader(path+"/app_categories.csv","CSV")
data_ca.index=data_ca.appID

d=data_ad.appID[data_train.creativeID]
d.index=data_train.index
data_train["appID"]=d

d=data_ad.appID[data_test.creativeID]
d.index=data_test.index
data_test["appID"]=d

d=data_ca["appCategory"][data_train.appID]
d.index=data_train.index
data_train["appCategoty"]=d

d=data_ca["appCategory"][data_test.appID]
d.index=data_test.index
data_test["appCategoty"]=d

#添加position
data_P=data_loader(path+"/position.csv","CSV")
data_P.index=data_P.positionID
d=data_P.loc[data_train.positionID]
d.index=data_train.index


del data_ad
del data_ca


data_one=data_train[data_train.label==1]
data_zero=data_train[data_train.label==0]

data_app=data_loader(path+"/user_installedapps.csv","CSV")
userID=data_app.userID.value_counts().index
data_train.to_csv("/home/ki/Desktop/pre/data_train.csv")
data_test.to_csv("/home/ki/Desktop/pre/data_test.csv")
"""

'''全为0.1
import pandas as pd
dd=pd.DataFrame({"instanceID":data_test.instanceID,"prob":data_test.instanceID.map(lambda x:0.1)})
'''
"""
#点击和转化的时间关系。
import datetime.datetime as dt

def int2time(x):
    return dt.strptime(str(x),"%d%H%M")
data_train.clickTime=data_train.clickTime.map(int2time)
dian=data_train.clickTime.value_counts()
zhuan=data_train.clickTime[data_train.label==1].value_counts()
dian.plot()
zhuan.plot()
#总体平稳除了19日和20日
#按照小时统计
data_train.clickTime=data_train.clickTime.map(lambda x:dt.datetime(x.year,x.month,x.day,x.hour))
dian=data_train.clickTime.value_counts()
zhuan=data_train.clickTime[data_train.label==1].value_counts()
dian.plot()
zhuan.plot()
(zhuan/dian).plot()#显然在不同的时段转化率不同
#按天统计
data_train.clickTime=data_train.clickTime.map(lambda x:dt.datetime(x.year,x.month,x.day))
dian=data_train.clickTime.value_counts()
zhuan=data_train.clickTime[data_train.label==1].value_counts()
(zhuan/dian).plot()#19,20,21天的数据比较奇怪。

#抽样查看500组用户的数据。
import datetime as dt
data_train=data_loader(path+"/train.csv","CSV")
def int2time(x):
    return dt.datetime.strptime(str(x),"%d%H%M")
data_train.clickTime=data_train.clickTime.map(int2time)
data_train["Time"]=data_train.clickTime.map(lambda x:dt.datetime(x.year,x.month,x.day))
import numpy as np
x=data_train.userID.value_counts().index
sp=[int(i*len(x)) for i in np.random.random(500)]
userPool=x[sp]
data_train.index=data_train.userID
pp=data_train.loc[userPool]
pp.clickTime.value_counts().plot()

#抽样50组APP查看。

import datetime as dt
data_train=data_loader(path+"/data_train.csv","CSV")
def int2time(x):
    return dt.datetime.strptime(str(x),"%d%H%M")
data_train.clickTime=data_train.clickTime.map(int2time)
data_train["Time"]=data_train.clickTime.map(lambda x:dt.datetime(x.year,x.month,x.day,x.hour))
import numpy as np
x=data_train.appID.value_counts().index
sp=[int(i*len(x)) for i in np.random.random(50)]
userPool=x[sp]
data_train.index=data_train.appID
pp=data_train.loc[userPool]
pp.index==list(range(len(pp)))
l=map(lambda x:pp.appID==x,userPool)
ll=pp.label==1
w=[[x[i] and j for i,j in ll.iteritems()] for x in l]
[pp[x].Time.value_counts().plot() for x in w]

#得到样本数大于1000的appID

appPool1=data_train.appID.value_counts()
appPool1=appPool1[appPool1>=1000].index

#得到按小时的转化率
appPool=data_train.appID.value_counts().index
data_train.index=data_train.appID
pp=appPool.map(lambda x:data_train.loc[x])
pp=pd.DataFrame(pp,index=appPool)
l=appPool.map(lambda x:pp.loc[x][0].Time.value_counts())
ll=appPool.map(lambda x:pp.loc[x][0][pp.loc[x][0].label==1].Time.value_counts())
n=len(appPool)
zhuan=[ll[i]/l[i]for i in range(n)]
def na_clear(data,method="zero"):
    if method=="zero":
        data[data.isnull()]=0
    if method=="inter":
        data.interpolate(methoed="value",inplace=True)
    if method=="time":
        data.interpolate(methoed="time",inplace=True)


[na_clear(i) for i in zhuan]


[zhuan[i].plot() for i in range(len(zhuan)) if pp.loc[appPool[i]][0].shape[0]>1000]

#检验positive和转化率的关系。
data_train.sort(["sitesetID","positionType"],inplace=True)
length=data_train.shape[0]
p=pd.Series(0,index=list(range(length)))
k=0
data_train.index=list(range(length))
pt=data_train.positionType
si=data_train.sitesetID
for i in data_train.index:
    if pt[i]==pt[i-1] and si[i]==si[i-1]:
        p[i]=k
    else:
        k+=1
        p[i]=k
data_train["position"]=p

positionPool=data_train.position.value_counts().index
data_train.index=data_train.position
pp=positionPool.map(lambda x:data_train.loc[x])
pp=pd.DataFrame(pp,index=positionPool)
l=positionPool.map(lambda x:pp.loc[x][0].Time.value_counts())
ll=positionPool.map(lambda x:pp.loc[x][0][pp.loc[x][0].label==1].Time.value_counts())
n=len(positionPool)
zhuan=[ll[i]/l[i] for i in range(n)]
[na_clear(i) for i in zhuan]
[zhuan[i].plot() for i in range(len(zhuan))]

#时间离散化
from datetime import *
data_train.Time=data_train.clickTime.map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
data_test.Time=data_test.clickTime.map(lambda x:datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

data_train.Time=data_train.Time.map(lambda x:x.hour)
data_test.Time=data_test.Time.map(lambda x:x.hour)

def lisan(x):
    if x>=23 or x<6:
        return 0
    if x>=6 and x<12:
        return 1
    if x>=12 and x<17:
        return 2
    else:
        return 3

#每日每个app的下载数
one=data_train[data_train.label==1]

data_UAA=data_loader(path+"/user_app_actioAns.csv","CSV")
data_UAA.installTime=data_UAA.installTime.map(lambda x:datetime.strptime(x,"%d%H%M"))
data_UAA["day"]=data_UAA.installTime.map(lambda x:x.day)

data_UAA.index=data_UAA.apply(lambda x:str(x["appID"])+"|"+str(x["day"]),axis=1)
APP_BY_DAY=data_UAA.index.valuecount()

one.conversionTime=one.conversionTime.map(lambda x:datetime.strptime(str(int(x)),"%d%H%M"))
one["day"]=one.conversionTime.map(lambda x:x.day)

one.index=one.apply(lambda x:str(x["appID"])+"|"+str(x["day"]),axis=1)
APP_BY_DAY.append(one.index.valuecount())
APP_BY_DAY.plot()
#类似的有CategoryByDay
ind=data_UI.userID.value_counts().index
data_UI.index=data_UI.userID
asd=ind.map(lambda x:data_UI.appID[x])
"""
#
def like(user,app,data_userIndex,data_appIndex):
    appPool=data_userIndex.appID.loc[user]
    userPool=appPool.map(lambda x:data_appIndex.userID.loc[x])
    u=data_appIndex.userID.loc[app]
    return ((userPool.map(lambda x:x.map(lambda y:x in u))).map(lambda x:sum(x)/len(x))).median()


data_CA=data_loader(path+"/app_categories.csv","CSV")
userPool=data_train.userID.value_counts().index
data_UAA=data_loader(path+"/user_app_actions.csv","csv")
#是否安装同类app，最近安装时间
def recent_install(user,appCa,data_userIndex):
    d=data_userIndex.loc[user]#data_UAA
    d.index=d.installedTime
    d=d.appCategory[d.appCategory==appCa]
    d.sort_index(ascending=False,inplace=True)
    return d.index[0]
data_UAA=data_UAA.merge(data_CA,on="appID")
d=pd.DataFrame(index=data_UAA.userID.value_counts().index,columns=data_UAA.appCategoty.value_counts().index,dtype=data_train.clickTime.dtype)
data_UAA.sort("installedTime",inplace=True)
def settime(x):
        d[x["appCategoty"]][x["userID"]]=x["clickTime"]
data_UAA.apply(settime,axis=1)
data_train["clickT"]=data_train.apply(lambda x:d[x["appCategoty"]][x["user"]],axis=1)
data_test["clickT"]=data_test.apply(lambda x:d[x["appCategoty"]][x["user"]],axis=1)
del data_UAA

#是否点击过同类app，点击次数
def clickTimes(user,appCa,data_userIndex):
    d=data_userIndex.appCategoty.loc[user]#data_train
    return d[d==appCa].shape[0]

d=pd.DataFrame(0,index=data_train.userID.value_counts().index,columns=data_train.appCategoty.value_counts().index)
def addone(x):
    d[x["appCategoty"]][x["userID"]]+=1

data_train.apply(addone,axis=1)
data_train["click"]=data_train.apply(lambda x:d[x["appCategoty"]][x["userID"]],axis=1)
data_test["click"]=data_test.apply(lambda x:d[x["appCategoty"]][x["userID"]] if x["userID"] in d.index else np.nan,axis=1)
#该用户的同类app点击转化率。

def clickzhuan(user,appCa,data_userIndex):
    d=data_userIndex.appCategoty.loc[user]#data_train
    return d[d==appCa].shape[0]/clickTimes(user,appCa)

one=data_train[data_train.label==1]
d=pd.DataFrame(0,index=data_train.userID.value_counts().index,columns=data_train.appCategoty.value_counts().index)
def addone(x):
    d[x["appCategoty"]][x["userID"]]+=1

one.apply(addone,axis=1)
data_train["zhuan"]=data_train.apply(lambda x:d[x["appCategoty"]][x["userID"]],axis=1)
data_train["zhuan"]=data_train["zhuan"]/data_train["click"]
data_test["zhuan"]=data_test.apply(lambda x:d[x["appCategoty"]][x["userID"]] if x["userID"] in d.index else np.nan,axis=1)
data_train["zhuan"]=data_test["zhuan"]/data_test["click"]
del data_CA
#是否点击过该app，点击次数
def clickSames(user,appID,data_userIndex):
    d = data_userIndex.appCategory.loc[user]#data_train
    return d[d == appID].shape[0]

d=pd.DataFrame(0,index=data_train.userID.value_counts().index,columns=data_train.appID.value_counts().index)
def addone(x):
    d[x["appID"]][x["userID"]]+=1
data_train.apply(addone,axis=1)
data_train["clickApp"]=data_train.apply(lambda x:d[x["appID"]][x["userID"]],axis=1)
data_test["clickApp"]=data_test.apply(lambda x:d[x["appID"]][x["userID"]],axis=1)
#是否安装相同app，最近安装时间
data_UAA=data_loader(path+"/user_app_actions.csv","csv")
def installSames(user,appID,data_userIndex):
    d = data_userIndex.loc[user]#user_UAA
    d.index = d.installedTime
    d = d.appID[d.appID == appID]
    d.sort_index(ascending=False, inplace=True)
    return d.index[0]

d=pd.DataFrame(index=data_UAA.userID.value_counts().index,columns=data_UAA.appID.value_counts().index,dtype=data_train.clickTime.dtype)
data_UAA.sort("installedTime",inplace=True)
def settime(x):
        d[x["appID"]][x["userID"]]=x["clickTime"]
one.apply(settime,axis=1)
data_train["zhuanT"]=data_train.apply(lambda x:d[x["appID"]][x["userID"]],axis=1)
data_test["zhuanT"]=data_test.apply(lambda x:d[x["appID"]][x["userID"]],axis=1)
del data_UAA
#用户对APP兴趣
def like(user,app,data_userIndex,data_appIndex):
    appPool=data_userIndex.appID.loc[user]#user_app
    userPool=appPool.map(lambda x:data_appIndex.userID.loc[x])
    u=data_appIndex.userID.loc[app]
    return ((userPool.map(lambda x:x.map(lambda y:x in u))).map(lambda x:sum(x)/len(x))).median()

#用户对广告接受度,用户该广告的点击次数
def guanggao(user,positiveID,data_userIndex):
    d = data_userIndex.positiveID.loc[user]#data_train
    return d[d == positiveID].shape[0]

C=data_train.column
def dian(col,one,data_train,data_test):
    d=data_train[col].value_counts()
    d.name="dian"+col
    d=pd.DataFrame(d)
    return (data_train.merge(d,left_on=col,right_index=True),data_test.merge(d,left_on=col,right_index=True))

for col in C:
    data_train,data_test=dian(col,one,data_train)

data_train.to_csv(path+"/data_train.csv",index=False)
data_test.to_csv(path + "/data_test.csv",index=False)

def zhuan(col,one,data_train,data_test):
    pool=data_train[col].value_counts()
    pooll=one[col].value_counts()
    d=pooll/pool
    d.name = "zhuan" + col
    d = pd.DataFrame(d)
    return (data_train.merge(d,left_on=col,right_index=True),data_test.merge(d,left_on=col,right_index=True))
for col in C:
    data_train,data_test=zhuan(col,one,data_train)
data_train.to_csv(path+"/data_train.csv")
data_test.to_csv(path + "/data_test.csv")

#特征选择～方差选择法
data_train.index=data_train.label
std=[np.std(data_train[i]/data_train[i].max()) for i in data_train.columns]
print(std)

import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import matplotlib.pylab as plt
import scipy as sp
def logloss(act,pred):
    epsilon=1e-15
    pred=sp.maximum(epsilon,pred)
    pred=sp.minimum(1-epsilon,pred)
    ll=sum(act*sp.log(pred)+sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll=ll*-1.0/len(act)
    return ll

def modelfit(alg, dtrain,dtest, dlabelx,dlabely, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()  # 获得参数列表
        xgtrain = xgb.DMatrix(dtrain.values, label=dlabelx.values)
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='logloss',
                          early_stopping_rounds=early_stopping_rounds
                          )
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain, dlabelx, eval_metric='auc')

    # Predict training set:
    dtest_predictions = alg.predict(dtest)
    dtest_predprob = alg.predict_proba(dtest)[:, 1]

    dtrain_predictions=alg.predict(dtrain)
    dtrain_predprob=alg.predict_proba(dtrain)[:,1]


    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" %metrics.accuracy_score(dlabely, dtest_predictions))
    print("AUC Score (Test): %f" %metrics.roc_auc_score(dlabely, dtest_predprob))

    print("Accuracy : %.4g" % metrics.accuracy_score(dlabelx, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dlabelx, dtrain_predprob))

    print("logloss Score (Train): %f" % logloss(dlabelx, dtrain_predprob))
    print("logloss Score (Test): %f" % logloss(dlabely, dtest_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def train(dtest,dtrain,dlabelx,dlabely,userTrainCV=True,cv_folds=5,early_stopping_rounds=50):

    param = {'max_depth': 10,#每棵树的最大深度
             'learning_rate': 0.1,#学习率
             'min_child_weight':5,#叶节点的最小权重和
             "max_delta_step":2,#最大迭代步长
             "subsample":0.1,#每次抽样比例
             "reg_alpha":0.1,#最小分裂
             "reg_lambda":300,
             "colsample_bytree":0.8,#?
             "seed":12,#随机数
             'silent': 0,#是否静默
             'objective': 'binary:logistic',#任务模式
             "n_estimators":1000,
             "scale_pos_weight":1
             }

    num_round=12#树规模
    watchlist=[(dtest, 'eval'), (dtrain, 'train')]#观察值
    xgb1=XGBClassifier(**param)
    modelfit(xgb1,dtrain,dtest,dlabelx,dlabely)
    data_train=xgb.DMatrix(dtrain,label=dlabelx)
def train2(data,label,train_rate=0.8):
    x=np.random.binomial(1,train_rate,size=data.shape[0])
    dtrain=xgb.DMatrix(data[x==1].values,label=label[x==1].values)
    dtest=xgb.DMatrix(data[x==0],label=label[x==0].values)
    label=xgb.DMatrix(label.values)

    param={
        "booster":"gbtree",
        "objective":"binary:logistic",
        "early_stopping_rounds":100,
        "scale_pos_weight":0.3,
        "eval_metric":"auc",
        "gamma":"0.1",
        "lambda":550,
        "subsample":0.7,
        "max_depth":8,
        "colsample_bytree":0.4,
        "min_child_weight":3,
        "eta":0.02,
        "seed":12,
    }
    clf=xgb.train(param,dtrain,num_boost_round=40,)
data_train=data_loader(path+"/data_train.csv","CSV")
data_test=data_loader(path+"/data_test.csv","CSV")
one=data_train[data_train.label==1]
zero=data_train[data_train.label==0]
w=data_train.label.map(lambda x:30 if x==1 else 1)
data=pd.concat([zero.sample(int(1.5*one.shape[0])),one])

dlabel=data.label
data.drop(["label","clickTime","conversionTime","userID","Unnamed: 0.1","Unnamed: 0"],axis=1,inplace=True)
data_test.drop(["label","clickTime","userID","Unnamed: 0"],axis=1,inplace=True)
train(data_test,data,dlabel=dlabel)

xxxx=np.random.binomial(1,0.8,size=data1.shape[0])

def cross(data,col1,col2):
    for i in col1:
        for j in col2:
            index1=data[i].value_counts().index
            index1=pd.Series(list(range(1,index1.shape[0]+1)),index=index1)

            index2 = data[j].value_counts().index
            index2 = pd.Series(list(range(1, index2.shape[0]+1)), index=index2)

            data[i+j]=data[[i,j]].apply(lambda x:index1[x[i]]*index2[x[j]]+index2[x[j]],axis=1)

appID=data.appID.value_counts().index
userPool=data.userID.value_counts().index


data_train.index=data_train.userID
data=data_loader(path+"/data.csv","CSV")
data=data[["label","userID","appID"]]
data["download"]=pd.DataFrame(0,index=data.index)
data_UI=data_loader(path+"/data_UI.csv","csv")
data_UI.index=data_UI.appID
data_UI[data_UI.appID.map(lambda x:x in appPool)]

recentUserApp=pd.concat([data_train,data_UAA])

for i in range(2642):
    if data_UI.iloc[i].userID in userPool:
        d = data.loc[data_UI.iloc[i].userID]
        if type(d) == pd.DataFrame:
            print(d)
            d = d[d.appID == data_UI.iloc[i].appID]
            d["download"] += 1
        elif d["appID"] == data_UI.iloc[i].appID:
            d["download"] += 1

for i in range(recentUserApp.shape[0]):
    x = recentUserApp.iloc[i]
    if x["userID"] in userPool:
        d = data.loc[x["userID"]]
        if type(d) == pd.DataFrame:
            d = d[d["time"] > x["time"]]
            appID = x["appID"]
            if type(d) == pd.DataFrame:
                d=d[d["appID"]==appID]
                d["recentDownload"] += 1
                d["download"] += 1
                d["click"] += 1
            elif d["appID"]==appID:
                d["recentDownload"] += 1
                d["download"] += 1
                d["click"] += 1
        elif d["time"] > x["time"] and d["appID"]==appID:
            d["recentDownload"] += 1
            d["download"] += 1
            d["click"] += 1
#click 未包含 download-recentDownload