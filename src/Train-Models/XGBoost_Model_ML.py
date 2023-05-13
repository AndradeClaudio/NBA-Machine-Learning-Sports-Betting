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

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values

data = data.astype(float)
acc_results = []
#acc_results.append(69.5)
acc_results_atual = []
#for x in tqdm(range(20), desc="Processando", ncols=80, leave=False):
for x in range(100):
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)
    #'eval_metric'=  mlogloss / merror  / auc
    param = {
        'max_depth': 9,
        'eta': 0.3,
        'gamma': 10,
        'max_delta_step': 5 ,
        'objective': 'multi:softprob',
        'num_class': 3 ,
        'eval_metric': 'mlogloss' ,
        'gpu_id' : 0 ,
        'tree_method' :'gpu_hist'
    }

    
    epochs = 500

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y)*100, 1)
    acc_results.append(acc)
    acc_results_atual.append(acc)
    print(f" {acc}% - Maxima: {max(acc_results)}%- Maxima Atual: {max(acc_results_atual)}%")
    # only save results if they are the best so far
    if acc == max(acc_results):
        model.save_model('../../Models/XGBoost_{}%_ML-2.json'.format(acc))
