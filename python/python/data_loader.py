import pandas as pd
import xlrd
from model import KNN
from model import predict
from math import log2
from functools import reduce
'''
作用：读取文件
:return pd.DataFrame
@path 文件路径.
@type 文件类型,csv:csv文件,excel:excel文件.
@header 选择列名行数,0:第0行,None:无标题,1:第一行,...,默认为0.
@encoding 编码格式,默认为UTF8.
@kw:sheet excel表格的页码,默认为0.
'''
def data_loader(path,type,header=0,encoding="utf8",sheet=0,na="",**kwargs):
    if type.lower()=="csv":
        return pd.read_csv(path,header=header,encoding=encoding,na_values=na)
    elif type.lower()=="excel":
        data=xlrd.open_workbook(path)
        table = data.sheet_by_index(sheet)
        list=[]
        for rownum in range(0,table.nrows):
            row = table.row_values(rownum)
            if row:
                col = []
                for i in range(table.ncols):
                    if row[i]==na:
                        col.append(None)
                    else:
                        col.append(row[i])
                list.append(col)

        if header:
            return pd.DataFrame(list[1:],columns=list[0])
        else:
            return pd.DataFrame(list)

    else:
        list = []
        with open(path,"r") as f:
            while True:
                list.append(f.readline().encode(encoding=encoding))
                if list[-1]==None or list[-1]=="":
                    break
        if header:
            return pd.DataFrame(list[1:], columns=list[0])
        else:
            return pd.DataFrame(list)

def nan_solver(data,col,NaNaction,all=0,**kwargs):

    if NaNaction.__class__ == str and col.__class__ in [str,int]:
        if all ==1:
            nan_solver(data,col=data.columns,NaNaction=[NaNaction for i in data.columns],**kwargs)
        else:
            if NaNaction=="N":
                pass
            elif NaNaction=="D":
                data.drop(data[data[col].isnull()].index,axis=0,inplace=True)
            elif NaNaction=="E":
                data[col][data[col].isnull()]=kwargs.get("value",0)
            elif NaNaction=="G":
                data[data[col].isnull()].map(lambda x:predict(kwargs.get("model",KNN(data,10,target=col)),target=col))
            else:
                print("Wrong!bad input of NaNaction.")
    elif "__getitem__" in dir(NaNaction) and "__getitem__" in dir(col):
        if len(NaNaction)!=len(col):
            print("Wrong input different length with col and NaNaction")
        else:
            for i in range(len(col)):
                nan_solver(data,col[i],NaNaction[i])
    else:
        print("Wrong!bad type of col and NaNaction.")


def next_binary(x,i=0,add=True):
    if add:
        if i == len(x):
            x[len(x)]=0
        if x[i]==1:
            x[i]=0
            return next_binary(x,i=i+1)
        else:
            x[i]=1
            return x
#binary循环待改进
def var_coder(data,col,codetype="binary",**kwargs):
    if "__iter__" not in dir(col):
        col=[col,]
    if codetype=="binary":
        data.sort(col,inplace=True)
        use=data[col]
        #统计总组合数
        num=1
        line=use.iloc[0]
        for i in use.index:
            ppp=(use.iloc[i] == line)

            if "__iter__" in dir(ppp) and reduce(lambda x,y:x and y,ppp) or ppp:
                line=use.iloc[i]
                num+=1
        #产生对应的binary序列
        length=int(log2(num))+1
        line=use.iloc[0]
        binaryline=pd.Series([0 for i in range(length)])
        d=pd.DataFrame(index=use.index,columns=kwargs.get("new_name",["var"+str(i) for i in range(length)]))
        for i in use.index:
            if not use.iloc[i].equals(line):
                line=use.iloc[i]
                binaryline=next_binary(binaryline)
                print(binaryline)

            for j in range(len(d.iloc[i])):
                d.iloc[i][j]=binaryline[j]

        data=data.drop(col,axis=1)
        return pd.concat([data,d],axis=1)

    if codetype=="single":
        data=data.copy()
        data_index=data.index
        for i in col:
            index=data[i].value_counts().index
            num=pd.Series(list(range(len(index))),index=index)
            data[col]=data[col].map(lambda x:num[x])

        return data
'''
path="/home/ki/Downloads/ccf_offline_stage1_train.csv"
data=data_loader(path,"CSV")
data=var_coder(data,"Date")
'''
def cut(array,breaks=10,ignore=1,step=None):
    if step==None:
        minI=min(array)
        maxI=max(array)*ignore
        m=minI*ignore-1
        M=maxI*ignore+1
        step=(M-m)/breaks
        print(type(breaks))
        l=[0 for i in range(breaks)]
        for i in range(1,breaks):
            l[i]=l[i-1]+step
        l[0]=minI-1
        l.append(maxI+1)

    return pd.Series([search(i) for i in array])


def search(x,l):
    start = 0
    end = len(l)-1

    while end - start > 1:
        median1 = int((start + end) / 2)
        median2 = median1 + 1
        if x >= l[median1] and x < l[median2]:
            start = median1
            end = median2
        elif x >= l[median2]:
            start = median2
        elif x < l[median1]:
            end = median1
    return str.format("{0}:{1}",l[start],l[end])


def var_discreter(data=None,col=None,array=None,method="cut",**kwargs):
    if data==None and array == None:
        print("No data imput")
    elif data!=None and array !=None:
        print("too many data input")
    elif array!=None:
        if method=="cut":
            return cut(array,breaks=kwargs.get("breaks",10))
        elif method=="by":
            return cutByY(array,kwargs.get("y"))
        elif method=="mutli":
            return cutByRandomTree(data,col,kwargs.get("y"))


