# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:22:19 2021

@author: wanghuishan
"""

"""
查看数据基本信息 数据大小，有哪些特征
哪些是数值特征，哪些是非数值特征
存在多少np.nan形式的缺失值
"""
from sklearn import preprocessing
class DataProcessing:   
    def preView(self,data):
        categoricalCols = []
        numericalCols = []
        print("data.shape",data.shape)
        print("data.columns",data.columns)
        # 非数值和数值型特征
        for col in data.columns:
            if data[col].dtype == 'object':
                categoricalCols.append(col)
            else:
                numericalCols.append(col)
    
        # 检查重复值
        dupNum = data.shape[0] - data.drop_duplicates().shape[0]
        print("数据集中有%s行重复值" % dupNum)
        
        # 查看非数值型特征缺失值
        # 删除缺失值达到60%以上的
        miss_cols = []
        for col in data.columns:
            missSum = data[col].isnull().sum()
            missRatio = 100*missSum/data.shape[0]
            if missRatio >= 60:
                data.drop(col,axis=1,inplace=True)
            elif missRatio>0:
                miss_cols.append(col)
                if col in numericalCols:
                    print("numericalCols：{} :缺失数 {} ,占比 :{:.1f}%".format(col,missSum,missRatio))
                else:
                    print("categoricalCols：{} :缺失数 {} ,占比 :{:.1f}%".format(col,missSum,missRatio))
        
        if miss_cols == []:
            print("None missing value")
            
        print("categoricalCols : ",len(categoricalCols))
        for col in categoricalCols:
            print(col + " unique value : %s" %data[col].unique().shape[0])
            
        print("numericalCols : ",len(numericalCols))
        for col in numericalCols:
            print(col)
            
        return numericalCols,categoricalCols

    """
    数据类型转换
    将Categorical数据LabelEncoder
    """
    def LabelEncoder(self,categoricalCols,data):
        encoder = preprocessing.LabelEncoder()
        for col in categoricalCols:
            data[col] = encoder.fit_transform(data[col])        
        return data