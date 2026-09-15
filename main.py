import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import math
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_squared_error

def ReadTrain_100plus():
    """读取单笔销售量大于100的训练集"""
    dataset = pd.read_csv(r'..\[new] yancheng_train_20171226.csv')
    dataset['sale_date'] = [int(i) for i in dataset['sale_date']]
    dataset = dataset.sort_index(axis=0, ascending=True, by='sale_date')

    train_x = dataset[(dataset['sale_date'] >= 201607) & (dataset['sale_date'] <= 201610)]
    train_x = train_x[train_x['sale_quantity'] > 100]
    train_y = dataset[(dataset['sale_date'] == 201611)]

    class_id_x = list(set(train_x['class_id']))
    class_id_y = list(set(train_y['class_id']))

    extra = []
    for y in class_id_y:
        if y not in class_id_x:
            extra.append(y)

    '去除掉没有在feature区间出现的class_id'
    for i in range(len(extra)):
        train_y = train_y[train_y['class_id'] != extra[i]]

    train_y = train_y[['class_id', 'sale_quantity']]
    train_y = train_y.groupby(by='class_id', as_index=False).sum()
    train_y = train_y.rename(columns={'sale_quantity': 'label'})

    return train_x, train_y

    pass
def ReadPre_100plus():
    """读取单笔销售量大于100预测集"""
    pre_x = pd.read_csv(r'..\[new] yancheng_train_20171226.csv')
    pre_x['sale_date'] = [int(i) for i in pre_x['sale_date']]
    pre_x = pre_x.sort_index(axis=0, ascending=True, by='sale_date')

    pre_x = pre_x[(pre_x['sale_date'] >= 201707) & (pre_x['sale_date'] <= 201710)]
    pre_x=pre_x[pre_x['sale_quantity']>100]

    pre_y = pd.read_csv(r'G:\Tianchi\yancheng\car_sales\row_data\yancheng_testA_20171225.csv')
    pre_y['predict_date'] = [int(i) for i in pre_y['predict_date']]

    return pre_x, pre_y

def ReadTrain_100minus():
    """读取单笔销售量小于或等于100的训练集"""
    dataset = pd.read_csv(r'..\[new] yancheng_train_20171226.csv')
    dataset['sale_date'] = [int(i) for i in dataset['sale_date']]
    dataset = dataset.sort_index(axis=0, ascending=True, by='sale_date')

    train_x = dataset[(dataset['sale_date'] >= 201607) & (dataset['sale_date'] <= 201610)]
    train_x = train_x[train_x['sale_quantity'] <= 100]
    train_y = dataset[(dataset['sale_date'] == 201611)]


    class_id_x = list(set(train_x['class_id']))
    class_id_y = list(set(train_y['class_id']))

    extra = []
    for y in class_id_y:
        if y not in class_id_x:
            extra.append(y)

    '去除掉没有在feature区间出现的class_id'
    for i in range(len(extra)):
        train_y = train_y[train_y['class_id'] != extra[i]]

    train_y = train_y[['class_id', 'sale_quantity']]
    train_y = train_y.groupby(by='class_id', as_index=False).sum()
    train_y = train_y.rename(columns={'sale_quantity': 'label'})

    return train_x, train_y

    pass
def ReadPre_100minus():
    """读取单笔销售量小于或等于100预测集"""
    pre_x = pd.read_csv(r'..\[new] yancheng_train_20171226.csv')
    pre_x['sale_date'] = [int(i) for i in pre_x['sale_date']]
    pre_x = pre_x.sort_index(axis=0, ascending=True, by='sale_date')

    pre_x = pre_x[(pre_x['sale_date'] >= 201707) & (pre_x['sale_date'] <= 201710)]
    pre_x=pre_x[pre_x['sale_quantity']<=100]

    pre_y = pd.read_csv(r'G:\Tianchi\yancheng\car_sales\row_data\yancheng_testA_20171225.csv')
    pre_y['predict_date'] = [int(i) for i in pre_y['predict_date']]

    return pre_x, pre_y

def XgbPre(train, pre):
    '预测'
    train_y = train['label']
    train_x = train.drop(['class_id', 'label'], axis=1)

    pre_index = pre[['predict_date', 'class_id', 'predict_quantity']]
    pre_x = pre.drop(['predict_date', 'class_id', 'predict_quantity'], axis=1)

    xgb_train = xgb.DMatrix(train_x.values, label=train_y.values)
    xgb_pre = xgb.DMatrix(pre_x.values)

    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eta': '0.01',
              'max_depth': 5,
              'eval_metric': 'rmse'
              }

    model = xgb.train(params, xgb_train, num_boost_round=1000)
    pre_index['predict_quantity'] = model.predict(xgb_pre)

    pass
def Ensemble(train, val):
    """Stacking模型融合"""
    train=train.fillna(0)
    val=val.fillna(0)
    train_y = train[['label']]
    train_x = train.drop(['class_id', 'label'], axis=1)

    if abs(train.columns.size - val.columns.size) != 0:
        tmp_zeros = pd.DataFrame(np.zeros((len(val), abs(train.columns.size - val.columns.size))))
        val = pd.concat([val, tmp_zeros], axis=1)

    val_class_id = val[['class_id']]
    val_quantity = val[['label']]
    val_x = val.drop(['class_id', 'label'], axis=1)
    print("特征数量",val_x.columns.size)
    gbdt = GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=3500)
    xgbr = xgb.XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, objective='reg:linear')
    lr = LinearRegression()
    rf = RandomForestRegressor(max_depth=5, n_estimators=100, criterion='mse')
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[gbdt, svr_rbf, lr, rf, xgbr], meta_regressor=xgbr)
    stregr.fit(train_x.values, train_y.values)
    stacking_result = stregr.predict(val_x.values)
    stacking_result = pd.DataFrame(stacking_result)

    '评价指标计算'
    mse = mean_squared_error(val_quantity.values, stacking_result.values)
    final_score = math.sqrt(mse)
    print("模型的rmse：", final_score)

    '画图'
    plt.scatter(range(len(val_quantity)), val_quantity['label'], label='True', color='r')
    plt.scatter(range(len(stacking_result)), stacking_result, label='Predict', color='b')
    plt.legend()
    plt.show()
    pass
	
if __name__=='__main__':
	pass
