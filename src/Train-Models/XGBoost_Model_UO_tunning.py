import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


dataset = "dataset_2012-23"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()
OU = data['OU-Cover']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover'], axis=1,
          inplace=True)
data = data.values
data = data.astype(float)
acc_results = []
acc_results.append(53.8)
#for x in tqdm(range(100)):
x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test)
cv_results = 0
# param = {'objective':'reg:squarederror', 
#          'max_depth': 8, 
#          'colsample_bylevel':0.5, 
#          'learning_rate':0.01, 
#          'random_state':20}
param = {'objective':'reg:squarederror', 
        'gpu_id' : 0 ,
        'tree_method' :'gpu_hist',
        'max_depth': 1,
        'eta': 0.08,
        'random_state':20
    }

cv_results = xgb.cv(dtrain=train, params=param, nfold=50, metrics={'rmse'}, as_pandas=True, seed=30, num_boost_round=1000)
print('RMSEa: %.2f' % cv_results['test-rmse-mean'].min())
## Result : RMSE: 2.69