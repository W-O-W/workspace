import pandas as pd
import xlrd
from model import KNN
from model import predict

'''
作用：读取文件
:return pd.DataFrame
@path 文件路径.
@type 文件类型,csv:csv文件,excel:excel文件.
@header 选择列名行数,0:第0行,None:无标题,1:第一行,...,默认为0.
@encoding 编码格式,默认为UTF8.
@kw:sheet excel表格的页码,默认为0.
'''
def data_loader(path,type,header=1,encoding="utf8",sheet=0,**kwargs):
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
                nan_solver(data,col[i],NaNaction)
    else:
        print("Wrong!bad type of col and NaNaction.")