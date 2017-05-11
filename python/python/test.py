'''
U假设检验
@x 数据X:pd.Series
@y 数据Y:pd.Series
@alpha 显著性水平:可选择0.05或者0.01,默认为0.05.
@dan 是否为单边检验:0~双边,1~单边.
@p 打印输出:0~不打印,1~打印,默认为打印.
'''
def U_test(x,y,alpha=0.05,p=1,dan=0):
    u=abs(x.mean() - y.mean()) / (x.var() / x.shape[0] + y.var() / y.shape[0]) ** 0.5
    if alpha == 0.05 and dan == 0:
        if u<=1.96:
            if p:
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
