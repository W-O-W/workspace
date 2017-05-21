from data_loader import *
path="/home/ki/Desktop/pre"
data_train=data_loader(path+"/train.csv","CSV")
data_test=data_loader(path+"/test.csv","CSV")
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



del data_ad
del data_ca


data_one=data_train[data_train.label==1]
data_zero=data_train[data_train.label==0]

data_app=data_loader(path+"/user_installedapps.csv","CSV")
userID=data_app.userID.value_counts().index
data_train.to_csv("/home/ki/Desktop/pre/data_train.csv")
data_test.to_csv("/home/ki/Desktop/pre/data_test.csv")

def get_simular(i,j,matrix):
    return matrix[i,j]/((matrix[i,i]*matrix[j,j])**0.5)

data_UAA=data_loader(path+"/user_app_actions.csv","CSV")
data_UAA.index=data_UAA.userID
ind=data_UAA.userID.value_counts().index
app=ind.map(lambda x:data_UAA.appID[x])
d={}
l=[ [d.update({i:d.get(i,{})}) for i in x] for x in app if type(x)!=type(app[-1])]#创建了空key
l=[[d[j].update({i:d.get(j).get(i,0)+1}) for j in d.keys() for i in x] for x in app]#添加值