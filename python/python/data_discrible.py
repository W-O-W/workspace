import pandas as pd
from ggplot import *
from numpy import nan
from test import U_test
from dmath import shannon
from dmath import dshannon

'''
对所选择的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@dtype 数据类型,continuous 或者 discrete,默认为continuous.
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，以及分布图,默认为means
@NaNaction 空缺值的处理方式,N:忽略,默认为忽略.
@p 打印输出:0~不打印,~打印,默认为打印.
'''
def feature_stater(data,colname,dtype="continuous",stat="means",NaNaction="n",p=1,**kwargs):
    if data.__class__==list:
        data=pd.DataFrame(data)
    if NaNaction.lower()=="n":
        if dtype.lower()=="continuous":
            return draw_ctn(data,colname,stat=stat,p=p,**kwargs)
        elif dtype.lower()=="discrete":
            return draw_dsct(data,colname,p=p,stat=stat)
    else:
        print("please convert to list or DataFrame")


'''
对连续类型的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，空缺值比例，以及分布图,默认为means
@p 打印输出:0~不打印,~打印,默认为打印.

'''
def draw_ctn(data,colname,stat="means",p=1,**kwargs):
    v = data[colname]
    if stat.lower()=="means":
        d = {}
        d["dtype"] = v.dtype
        d["max"] = v.max()
        d["min"] = v.min()
        d["mean"] = v.mean()
        d["std"] = v.std()
        d["skew"]=v.skew()
        d["kurt"]=v.kurt()
        valueNull=v.isnull().value_counts()
        if True in valueNull.index:
            d["NA"]=valueNull[True]
        else:
            d["NA"]=0
        d["NArate"]=d["NA"]/v.shape[0]
        if p:
            print(d)
            print(ggplot(data, aes(x=colname)) + geom_bar(binwidth=kwargs.get("binwidth", (d["max"] - d["min"]) / 10),stat="identity"))
        return d


'''
对离散类型的列进行描述性统计
:return dict
@data 数据,格式要求列表或者DataFrame
@colname 要统计的列名
@stat 统计内容,means:基本统计包含 最大值，最小值，均值，标准差，空缺值频数，以及分布图,默认为means
@p 打印输出:0~不打印,1~打印,默认为打印.

'''
def draw_dsct(data, colname,stat="means",p=1):
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
        d["skew"]=vount.skew()
        d["kurt"]=vount.kurt()

        valueNull=v.isnull().value_counts()
        if True in valueNull.index:
            d["NA"]=valueNull[True]
        else:
            d["NA"]=0
        d["NArate"]=d["NA"]/v.shape[0]

        if p:
            print(d)
            print(ggplot(data, aes(x=colname)) + geom_bar(stat="bin"))
        return d

#0单边，1双边


'''
对所选择的列进行描述性统计
:return three dict dx~x的统计值,dy~y的统计值,dxy~xy的统计值
@data 数据:DataFrame.
@xlabel,ylabel 要比较的两个列列名:String/int.
@xtype,ytype 对应两列的数据类型:"continuous" 或者 "discrete",默认为"continuous".
@stat 统计内容:"means"~基本统计包含 最大值，最小值，均值，标准差，空缺值频数，以及分布图,默认为"means".
@NaNaction 空缺值的处理方式:"N"~忽略,默认为忽略.
@p 打印输出:0~不打印,1~打印,默认为1.

'''
def feature_discribler(data,xlabel,ylabel=0,xtype="continuous",ytype="continuous",stat="means",NaNaction="n",p=1,**kwargs):
    if data.__class__==list:
        data=pd.DataFrame(data)
    if NaNaction.lower()=="n":
        #查看两个变量的基本情况。
        dx=feature_stater(data, xlabel, xtype, stat, NaNaction,p=p,**kwargs)
        input()
        dy=feature_stater(data,ylabel,ytype,stat,NaNaction,p=p,**kwargs)
        input()
        p=ggplot(data,aes(x=xlabel,y=ylabel))
        if xtype.lower()=="continuous" and ytype.lower()=="continuous":
            dxy={}
            dxy["corr"]=data[xlabel].corr(data[ylabel],method=kwargs.get("corr_method","pearson"))
            if p:
                print(dxy)
                print(p+geom_point())
        if xtype.lower()=="discrete" and ytype.lower()=="discrete":
            dxy={}

            dx["shanon"]=shannon(data[xlabel],**kwargs)
            dy["shanon"]=shannon(data[ylabel],**kwargs)
            dxy["ySx"]=dshannon(data,ylabel,xlabel)
            dxy["xSy"]=dshannon(data,xlabel,ylabel)

            if p:
                print(dxy)
                print(p+geom_point())

        if xtype.lower()=="discrete" and ytype.lower()=="continuous":
            dxy={}
            dx["shannon"]=shannon(data[xlabel],**kwargs)
            #显著性检验,U检验
            split_by_x=data[xlabel].value_counts().index.map(lambda x: data[ylabel][data[xlabel] == x])
            dxy["xUy"]=split_by_x.map(lambda x:U_test(x,data[xlabel]))

            if p:
                print(dxy)
                print(p+geom_boxplot())

        if ytype.lower()=="continuous" and ytype.lower()=="discrete":
            dxy={}
            dy["shannon"]=shannon(data[ytype])
            #显著性检验,U检验
            split_by_y = data[ylabel].value_counts().index.map(lambda x: data[xlabel][data[ylabel] == x])
            dxy["yUx"] = split_by_y.map(lambda x: U_test(x, data[ylabel]))

            if p:
                print(dxy)
                print(p+geom_point())
    else:
        print("please convert to list or DataFrame")


def data_discribler(data,head=5,tail=5,corr_method="pearson",NaNaction="N"):
    print("data's head:")
    print(data.head(head))

    print("data's tail")
    print(data.tail(tail))

    print("cor matrix of nson")
    corr=data.corr(corr_method)
    if nan in corr:
        print("Wrong!! nan in data")
        print(corr)



