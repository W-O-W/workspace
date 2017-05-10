import pandas as pd
import xlrd
import numpy as np
from ggplot import *
from math import log
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
def load_data(path,type,header=0,encoding="utf8",sheet=0,**kwargs):
    if type.lower()=="csv":
        return pd.read_csv(path,header=header,encoding=encoding)
    elif type.lower()=="excel":
        data=xlrd.open_workbook(path)
        table = data.sheet_by_index(sheet)
        list=[]
        for rownum in range(0,table.nrows):
            row = table.row_values(rownum)
            if row:
                col = []
                for i in range(table.ncols):
                    if row[i]=="":
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


'''
对所选择的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@dtype 数据类型,continuous 或者 discrete,默认为continuous.
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，以及分布图,默认为means
@NaNaction 空缺值的处理方式,N:忽略,默认为忽略。
'''
def feature_stat(data,colname,dtype="continuous",stat="means",NaNaction="n",**kwargs):
    if data.__class__==list:
        data=pd.DataFrame(data)
    if NaNaction.lower()=="n":
        if dtype.lower()=="continuous":
            return draw_ctn(data,colname,stat,**kwargs)
        elif dtype.lower()=="discrete":
            return draw_dsct(data,colname,stat)
    else:
        print("please convert to list or DataFrame")


'''
对连续类型的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，空缺值比例，以及分布图,默认为means
'''
def draw_ctn(data,colname,stat,**kwargs):
    v = data[colname]
    if stat.lower()=="means":
        d = {}
        d["dtype"] = v.dtype
        d["max"] = v.max()
        d["min"] = v.min()
        d["mean"] = v.mean()
        d["std"] = v.std()
        valueNull=v.isnull().value_counts()
        if True in valueNull.index:
            d["NA"]=valueNull[True]
        else:
            d["NA"]=0
        d["NArate"]=d["NA"]/v.shape[0]

        print(d)
        print(ggplot(data, aes(x=colname)) + geom_bar(binwidth=kwargs.get("binwidth", (d["max"] - d["min"]) / 10),stat="identity"))
        return d


'''
对离散类型的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，以及分布图,默认为means
'''
def draw_dsct(data, colname, stat):
    v=data[colname]
    if stat.lower()=="means":
        vount = v.value_counts()
        d = {}
        d["dtype"] = v.dtype
        d["maxId"] = vount.idxmax()
        d["max"]=vount.max()
        d["minId"]=vount.idxmin()
        d["min"] = vount.min()
        d["mean"] = vount.mean()
        d["std"] = vount.std()
        valueNull=v.isnull().value_counts()
        if True in valueNull.index:
            d["NA"]=valueNull[True]
        else:
            d["NA"]=0
        d["NArate"]=d["NA"]/v.shape[0]

        print(d)
        print(ggplot(data, aes(x=colname)) + geom_bar(stat="bin"))
        return d


def shannon(vec,**kwargs):
    reduce(lambda y, z: y + z * log(z, kwargs.get("shanon_base", 2)),
           vec.value_counts().map(lambda x: x / vec.shape[0]))
def dshannon(data,xlabel,ylabel,**kwargs):
           data[xlabel].value_counts().index.map(lambda x: shannon(data[ylabel][data[xlabel] == x]),**kwargs)
#0单边，1双边
def U_test(x,y,alpha=0.05,dan=0):
    u=abs(x.mean() - y.mean()) / (x.var() / x.shape[0] + y.var() / y.shape[0]) ** 0.5
    if alpha == 0.05 and dan == 0:
        if u<=1.96:
            print("无显著性查别，u=",u,"alpha=0.05")
            return False
        else:
            print("有显著性查别，u=",u,"alpha=0.05")
            return True

    if alpha == 0.05 and dan == 1:
        if u<=1.6449:
            print("无显著性查别，u=",u,"alpha=0.05")
            return False
        else:
            print("有显著性查别，u=",u,"alpha=0.05")
            return True
    if alpha == 0.01 and dan == 0:
        if u<=3.29:
            print("无显著性查别，u=",u,"alpha=0.05")
            return False
        else:
            print("有显著性查别，u=",u,"alpha=0.05")
            return True
    if alpha == 0.01 and dan == 1:
        if u<=3.09:
            print("无显著性查别，u=",u,"alpha=0.05")
            return False
        else:
            print("有显著性查别，u=",u,"alpha=0.05")
            return True


def feature_discrible(data,xlabel,ylabel=0,xtype="continuous",ytype="continuous",stat="means",NaNaction="n",**kwargs):
    if data.__class__==list:
        data=pd.DataFrame(data)
    if NaNaction.lower()=="n":
        #查看两个变量的基本情况。
        dx=feature_stat(data, xlabel, xtype, stat, NaNaction,**kwargs)
        input()
        dy=feature_stat(data,ylabel,ytype,stat,NaNaction,**kwargs)
        input()
        p=ggplot(data,aes(x=xlabel,y=ylabel))
        if xtype.lower()=="continuous" and ytype.lower()=="continuous":
            dxy={}
            dxy["corr"]=data[xlabel].corr(data[ylabel],method=kwargs.get("corr_method","pearson"))

            print(dxy)
            print(p+geom_point())
        if xtype.lower()=="discrete" and ytype.lower()=="discrete":
            dxy={}

            dx["shanon"]=shannon(data[xlabel],**kwargs)
            dy["shanon"]=shannon(data[ylabel],**kwargs)
            dxy["ySx"]=dshannon(data,ylabel,xlabel)
            dxy["xSy"]=dshannon(data,xlabel,ylabel)

            print(dxy)
            print(p+geom_point())

        if xtype.lower()=="discrete" and ytype.lower()=="continuous":
            dxy={}
            dx["shannon"]=shannon(data[xlabel],**kwargs)
            #显著性检验,U检验
            split_by_x=data[xlabel].value_counts().index.map(lambda x: data[ylabel][data[xlabel] == x])
            dxy["xUy"]=split_by_x.map(lambda x:U_test(x,data[xlabel]))

            print(dxy)
            print(p+geom_boxplot())

        if ytype.lower()=="continuous" and ytype.lower()=="discrete":
            dxy={}
            dy["shannon"]=shannon(data[ytype])
            #显著性检验,U检验
            split_by_y = data[ylabel].value_counts().index.map(lambda x: data[xlabel][data[ylabel] == x])
            dxy["yUx"] = split_by_y.map(lambda x: U_test(x, data[ylabel]))

            print(dxy)
            print(p+geom_point())
    else:
        print("please convert to list or DataFrame")





