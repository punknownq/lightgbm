import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score

df_train = pd.read_table("kddtrain2018.txt", header=None, sep=" ")
df_test = pd.read_table("kddtest2018.txt", header=None, sep=" ")
# 获取训练数据
data_x = df_train[list(range(100))].values
# test_x = df_test[list(range(100))]
# 获取训练label
data_y = df_train.iloc[:, -1].values
# test_y = df_test.icol(-1)
X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.1)
train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':4,
    'num_leaves': 16,
    'objective':'multiclass',
    'num_class':3,
    'verbose': -1
}
#leaning_rate0.1
#0.9904306220095693
#leaning_rate0.03
#0.9665071770334929
#leaning_rate0.03
#0.9346092503987241
#Best situation
# 'learning_rate':0.1,
#     'lambda_l1':0.1,
#     'lambda_l2':0.2,
#     'max_depth':4,
#     'num_leaves': 16,
#     'objective':'multiclass',
#     'num_class':3,
#     'verbose': -1
# 0.9952153110047847
clf=lgb.train(params,train_data,valid_sets=[validation_data])
y_pred=clf.predict(X_test)
y_pred=[list(x).index(max(x)) for x in y_pred]
#print(y_pred)
print(accuracy_score(y_test,y_pred))