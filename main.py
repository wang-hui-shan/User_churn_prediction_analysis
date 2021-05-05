# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:56:49 2021

@author: wanghuishan

"""
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

train_data = pd.read_csv('datasets/Telco-Customer-Churn.csv')
"""
21列原始属性中，除了最后一列Churn表示该数据集的目标变量（Churn）外
其余20列按照原始数据集中的排列顺序刚好可以分为三类特征群：
即客户的基本信息(customerID-tenure)、开通业务信息(PhoneService-StreamingMovies)、签署的合约信息(Contract-TotalCharges)。
"""

pd.set_option('display.max_columns', None)    # 显示所有列
train_data.head(10)

# 数据集中就有这样一列TotalCharges特征，存在样本，其特征值为空格字符（' '）
# convert_numeric如果为True，则尝试强制转换为数字，不可转换的变为NaN

# train_data[train_data['TotalCharges']==' '][['TotalCharges','MonthlyCharges','Churn']]
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'],errors='coerce')

from MyUtils import DataProcessing
# 类需要实例化
DataProcessing = DataProcessing()
# 只检查np.nan形式的缺失值
numericalCols,categoricalCols = DataProcessing.preView(data=train_data)

"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='median')
imputer.fit_transform()
更进一步的，可以发现缺失样本中tenure特征（表示客户的入网时间）均为0，
且在整个数据集中tenure为0与TotalCharges为缺失值是一一对应的。
结合实际业务分析，这些样本对应的客户可能入网当月就流失了，
但仍然要收取当月的费用，因此总费用即为该用户的每月费用（MonthlyCharges）。
因此本案例最终采用MonthlyCharges的数值对TotalCharges进行填充。
"""
# 填充缺失值
train_data['TotalCharges'] = train_data['TotalCharges'].fillna(train_data['MonthlyCharges'])

"""
Tableaud数据可视化分析
什么样的客户在流失?（流失客户占比26.54%）
基本信息：
客户流失和性别没有太大关系，男女流失率差不多。
流失客户大部分入网时间不长。
老年客户的占总客户比例小，但是流失率最高，达到了40%。
没有亲人的独居客户流失率较大。

业务信息：
业务办理越少的客户流失率越大。

合约信息：
退网用户多为短期（月期）合约，并且付款方式多采用电子支票。
月支出在70-100消费之间的流失率较高
初入网用户不稳定，流失率大。
"""
"""
import seaborn as sns
import matplotlib.pyplot as plt
# 月支出核密度分布函数
# 月支出在70-100消费之间的流失率较高
plt.figure()
sns.displot(x='MonthlyCharges',hue = 'Churn',data = train_data,kind='kde')
plt.show()

# 总支出核密度分布函数
plt.figure()
sns.displot(x='TotalCharges',hue = 'Churn',data = train_data,kind='kde')
plt.show()
"""

"""
特征工程：特征工程利用数据领域的相关知识来创建能够使机器学习算法达到最佳性能的特征。
1.特征构建 ：从原始数据中人工的构建新的特征。
对于MonthlyCharges，TotalCharges两个连续特征，构建两个分桶特征量。
2.特征提取 ：动地构建新的特征，将原始特征转换为一组具有明显物理意义或者统计意义或核的特征。
3.特征选择 ：从特征集合中挑选一组最具统计意义的特征子集，从而达到降维的效果。
"""

# 构建MonthlyChargesBin， TotalChargesBin特征
bins = (train_data['tenure'].max()-train_data['tenure'].min())/4
bins = int(round(bins,0))
train_data['tenureBin'] = pd.cut(train_data['tenure'],bins)

bins = (train_data['MonthlyCharges'].max()-train_data['MonthlyCharges'].min())/5
bins = int(round(bins,0))
train_data['MonthlyChargesBin'] = pd.cut(train_data['MonthlyCharges'],bins)

bins = (train_data['TotalCharges'].max()-train_data['TotalCharges'].min())/200
bins = int(round(bins,0))
train_data['TotalChargesBin'] = pd.cut(train_data['TotalCharges'],bins)

categoricalCols.append('tenureBin')
categoricalCols.append('MonthlyChargesBin')
categoricalCols.append('TotalChargesBin')


# 类别特征编码 首先将部分特征值进行合并
'''
train_data.loc[train_data['MultipleLines']=='No phone service', 'MultipleLines'] = 'No'

internetCols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in internetCols:
    train_data.loc[train_data[i]=='No internet service', i] = 'No'
'''
train_data = DataProcessing.LabelEncoder(categoricalCols,train_data)

# 相关系数
#abs(train_data.corr()['Churn']).sort_values(ascending=False)

"""
结合可视分析和相关性分析删除特征
StreamingMovies      0.038492
MultipleLines        0.038037
StreamingTV          0.036581
customerID           0.017447
PhoneService         0.011942
gender               0.008612
"""
# 删除特征
drop_cols = ['StreamingMovies','MultipleLines','StreamingTV','customerID','PhoneService','gender']
train_data.drop(columns=drop_cols,axis=1,inplace=True)

# SMOTE 上采样平衡样本
# 平衡前后评价指标变化明显，不知道是不是过拟合了
from imblearn.over_sampling import SMOTE

train_X = train_data.drop(['Churn'],axis=1)
train_y = train_data[['Churn']]

X_resampled, y_resampled = SMOTE().fit_resample(train_X, train_y)

# 模型构建
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,precision_score, recall_score, f1_score    # 导入精确率、召回率、F1值等评价指标

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# 编码的特征不需要标准化
scaler = StandardScaler()
X_resampled[['tenure']] = scaler.fit_transform(X_resampled[['tenure']])
X_resampled[['MonthlyCharges']] = scaler.fit_transform(X_resampled[['MonthlyCharges']])
X_resampled[['TotalCharges']] = scaler.fit_transform(X_resampled[['TotalCharges']])

# 未调参
lr = LogisticRegression()
svc = SVC()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
xgb = XGBClassifier()

scores = {'precision': make_scorer(precision_score), 
          'recall': make_scorer(recall_score),
          'f1':make_scorer(f1_score)}

lr_cv_results = cross_validate(lr,X_resampled,y_resampled,scoring=scores)

svc_cv_results = cross_validate(svc,X_resampled,y_resampled,scoring=scores)

rfc_cv_results = cross_validate(rfc,X_resampled,y_resampled,scoring=scores)

gbc_cv_results = cross_validate(gbc,X_resampled,y_resampled,scoring=scores)

xgb_cv_results = cross_validate(xgb,X_resampled,y_resampled,scoring=scores)

scoreDf = pd.DataFrame(columns=['LogisticRegression', 'SVC', 'RandomForest', 'GradientBoosting', 'XGBoost'])
cv_results = [lr_cv_results, svc_cv_results, rfc_cv_results, gbc_cv_results, xgb_cv_results]

for i in range(len(cv_results)):
    scoreDf.iloc[:, i] = pd.Series([cv_results[i]['test_recall'].mean(),
                                   cv_results[i]['test_precision'].mean(),
                                   cv_results[i]['test_f1'].mean()])

scoreDf.index = ['Recall_mean', 'Precision_mean', 'F1-score_mean']
print(scoreDf)


# 部分调参
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# C
lr = LogisticRegression(C=0.01)
lr_cv_results = cross_validate(lr,X_resampled,y_resampled,scoring=scores)

svc = SVC(C=0.01,kernel='poly')
# C kernel ['linear', 'poly', 'rbf', 'sigmoid']
svc_cv_results = cross_validate(svc,X_resampled,y_resampled,scoring=scores)

'''
n_estimators [1,200]
max_depth min_samples_leaf min_samples_split [1,20] np.arange(1,20,1)

test_recall_scores = []
for i in np.arange(80,100,1):
    cv_results = cross_validate(RandomForestClassifier(n_estimators=i),X_resampled,y_resampled,scoring=scores)
    test_recall_scores.append(cv_results['test_recall'].mean())
'''

    
rfc = RandomForestClassifier(n_estimators=85,max_depth=2,min_samples_leaf=10,min_samples_split=3)
rfc_cv_results = cross_validate(rfc,X_resampled,y_resampled,scoring=scores)

# n_estimators 
# subsample
gbdt = GradientBoostingClassifier(n_estimators=7)
gbdt_cv_results = cross_validate(gbdt,X_resampled,y_resampled,scoring=scores)  

xgb = XGBClassifier(n_estimators=8)
xgb_cv_results = cross_validate(xgb,X_resampled,y_resampled,scoring=scores)

CV_scoreDf = pd.DataFrame(columns=['LogisticRegression', 'SVC', 'RandomForest', 'GradientBoosting', 'XGBoost'])
cv_results = [lr_cv_results, svc_cv_results, rfc_cv_results, gbdt_cv_results, xgb_cv_results]

for i in range(len(cv_results)):
    CV_scoreDf.iloc[:, i] = pd.Series([cv_results[i]['test_recall'].mean(),
                                   cv_results[i]['test_precision'].mean(),
                                   cv_results[i]['test_f1'].mean()])

CV_scoreDf.index = ['Recall_mean', 'Precision_mean', 'F1-score_mean']
print(CV_scoreDf)

# 在模型预测阶段，可以结合预测出的概率值决定对哪些客户进行重点留存：
svc = SVC(C=0.01,kernel='poly',probability=True)
svc = svc.fit(X_resampled,y_resampled)
prob_y = svc.predict_proba(X_resampled)

# 流失的概率
prob_y[:,1]
# 流失概率大于0.6的客户
prob_y[prob_y[:,1]>0.6,1]