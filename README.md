# User_churn_prediction_analysis
电信用户流失预测分析
| $变量名$ | $描述$ | $数据类型$ | $取值$ | $所属特征群或标签$ |
| :-----:| :----: | :----: | :----: | :----: |
| customerID | 客户ID | 字符串 | 7043个不重复取值 | 基本信息 |
| gender | 性别 | 字符串 | Male, Female | 基本信息 |
| SeniorCitizen | 是否为老年人 | 整型 | 1, 0 | 基本信息 |
| Partner | 是否有配偶 | 字符串 | Yes, No | 基本信息 |
| Dependents | 是否有家属 | 字符串 | Yes, No | 基本信息 |
| tenure | 入网月数 | 整型 | 0～72 | 基本信息 |
| PhoneService | 是否开通电话业务 | 字符串 | Yes, No | 开通业务信息 |
| MultipleLines | 是否开通多线业务 | 字符串 | Yes, No, No phone service | 开通业务信息 |
| InternetService | 是否开通互联网业务 | 字符串 | DSL数字网络, Fiber optic光纤网络, No | 开通业务信息 |
| OnlineSecurity | 是否开通在线安全业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| OnlineBackup | 是否开通在线备份业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| DeviceProtection | 是否开通设备保护业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| TechSupport | 是否开通技术支持业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| StreamingTV | 是否开通网络电视业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| StreamingMovies | 是否开通网络电影业务 | 字符串 | Yes, No, No internet service | 开通业务信息 |
| Contract | 合约期限 | 字符串 | Month-to-month, One year, Two year | 签署的合约信息 |
| PaperlessBilling | 是否采用电子结算 | 字符串 | Yes, No | 签署的合约信息 |
| PaymentMethod | 付款方式 | 字符串 | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | 签署的合约信息 |
| MonthlyCharges | 每月费用 | 浮点型 | 18.25～118.75 | 签署的合约信息 |
| TotalCharges | 总费用 | 字符串 | 有部分空格字符，除此之外的字符串对应的浮点数取值范围在18.80～8684.80之间 | 签署的合约信息 |
| Churn | 客户是否流失 | 字符串 | Yes, No | 目标变量 |
