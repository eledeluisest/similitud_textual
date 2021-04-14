# Entrenamiento de un modelo de ML
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

import pickle

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

df_train = pd.read_csv("data/train_features.csv", sep=';')
df_dev = pd.read_csv("data/dev_features.csv", sep=';')
df_test = pd.read_csv("data/test_features.csv", sep=';')



# Baseline
corr_baseline_train = df_train[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_dev = df_dev[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_test = df_test[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
print(corr_baseline_train, corr_baseline_dev, corr_baseline_test)

# Conjuntos de train, dev y test

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]

y_test = df_test['num2']
x_test = df_test[df_test.columns[15:]]


modelos = ["res/MLP_model.sav", "res/SVR_model.sav", "res/lgbm_model.sav", "res/rf_model.sav", "res/lr_model.sav"]


df_train_all = df_train
df_dev_all = df_dev
df_test_all = df_test
for mod in modelos:
    with open(mod, 'rb') as f:
        estimador = pickle.load(f)
    print(estimador.best_estimator_)

    print(mod+" train", np.corrcoef(y_train, estimador.predict(x_train))[0, 1])
    print(mod+" dev", np.corrcoef(y_dev, estimador.predict(x_dev))[0, 1])
    print(mod+" test", np.corrcoef(y_test, estimador.predict(x_test))[0, 1])

    df_train_all = pd.concat([df_train_all, pd.Series(estimador.predict(x_train)).rename(mod.split('_')[0].split('/')[1])], axis=1)
    df_dev_all = pd.concat([df_dev_all, pd.Series(estimador.predict(x_dev)).rename(mod.split('_')[0].split('/')[1])], axis=1)
    df_test_all = pd.concat([df_test_all, pd.Series(estimador.predict(x_test)).rename(mod.split('_')[0].split('/')[1])], axis=1)

df_train_all['final_score'] =  df_train_all.iloc[:,-5:].apply(sum, axis=1)
df_dev_all['final_score'] =  df_dev_all.iloc[:,-5:].apply(sum, axis=1)
df_test_all['final_score'] =  df_test_all.iloc[:,-5:].apply(sum, axis=1)

modelos = df_train_all.columns[-6:].values
list_all = []
list_news = []
list_captions = []
list_forums = []
for model in modelos:
    list_all.append([model, df_train_all[['num2', model]].corr().iloc[0,1],
                     df_dev_all[['num2', model]].corr().iloc[0,1],
                     df_test_all[['num2', model]].corr().iloc[0,1]])
    list_news.append([model, df_train_all[df_train_all['genero'] == 'main-news'][['num2', model]].corr().iloc[0,1],
                     df_dev_all[df_dev_all['genero'] == 'main-news'][['num2', model]].corr().iloc[0,1],
                     df_test_all[df_test_all['genero'] == 'main-news'][['num2', model]].corr().iloc[0,1]])
    list_captions.append([model, df_train_all[df_train_all['genero'] == 'main-captions'][['num2', model]].corr().iloc[0,1],
                     df_dev_all[df_dev_all['genero'] == 'main-news'][['num2', model]].corr().iloc[0,1],
                     df_test_all[df_test_all['genero'] == 'main-news'][['num2', model]].corr().iloc[0,1]])
    list_forums.append([model, df_train_all[df_train_all['genero'] == 'main-forum'][['num2', model]].corr().iloc[0,1],
                     df_dev_all[df_dev_all['genero'] == 'main-forums'][['num2', model]].corr().iloc[0,1],
                     df_test_all[df_test_all['genero'] == 'main-forums'][['num2', model]].corr().iloc[0,1]])


df_res_all = pd.DataFrame(list_all, columns=['Modelo', 'Train', 'Dev', 'Test']).set_index('Modelo')

df_res_news = pd.DataFrame(list_news, columns=['Modelo', 'Train_news', 'Dev_news', 'Test_news']).set_index('Modelo')

df_res_captions = pd.DataFrame(list_captions, columns=['Modelo', 'Train_captions', 'Dev_captions', 'Test_captions']).set_index('Modelo')

df_res_forums = pd.DataFrame(list_forums, columns=['Modelo', 'Train_forums', 'Dev_forums', 'Test_forums']).set_index('Modelo')


pd.concat([df_res_all, df_res_news, df_res_captions, df_res_forums], axis=1).to_csv("res/resultados_modelos.csv", sep=";", decimal=",")