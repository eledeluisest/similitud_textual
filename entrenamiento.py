# Entrenamiento de un modelo de ML
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

import pickle
from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

df_train = pd.read_csv("data/train_features.csv", sep=';')
df_dev = pd.read_csv("data/dev_features.csv", sep=';')
df_test = pd.read_csv("data/test_features.csv", sep=';')
# Base line -> Mejor métrica y combinación de mejores métricas para cada bloque
# Mejor metrica
corr_baseline_train = df_train[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_dev = df_dev[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_test = df_test[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
print(corr_baseline_train, corr_baseline_dev, corr_baseline_test)
# Mejor metrica por bloque -> La conclusion es que no funciona bien porque las diferentes métricas se distribuyen de
# Forma muy diferente y afecta mucho al cálculo de la correlación.
"""
max_train_mih = df_train['nosw_score_milha_vw_cosin'].max()
min_train_mih = df_train['nosw_score_milha_vw_cosin'].min()
df_train['nosw_score_milha_vw_cosin_norm'] = (df_train['nosw_score_milha_vw_cosin'] - min_train_mih) / (
            max_train_mih - min_train_mih)

max_train_ali = df_train['nosw_score_aligned'].max()
min_train_ali = df_train['nosw_score_aligned'].min()
df_train['nosw_score_aligned_norm'] = (df_train['nosw_score_aligned'] - min_train_ali) / (
            max_train_ali - min_train_ali)

df_train.loc[df_train['genero'] == 'main-news', "best_bloques"] = df_train.loc[
    df_train['genero'] == 'main-news', "nosw_score_aligned_norm"]

df_train.loc[df_train['genero'] == 'main-captions', "best_bloques"] = df_train.loc[
    df_train['genero'] == 'main-captions', "nosw_score_milha_vw_cosin_norm"]

df_train.loc[df_train['genero'] == 'main-forum', "best_bloques"] = df_train.loc[
    df_train['genero'] == 'main-forum', "nosw_score_aligned_norm"]

corr_best_train = df_train[['num2', 'best_bloques']].corr().loc["best_bloques", "num2"]

df_dev.loc[df_dev['genero'] == 'main-news', "best_bloques"] = df_dev.loc[
    df_dev['genero'] == 'main-news', "nosw_score_aligned"]

df_dev.loc[df_dev['genero'] == 'main-captions', "best_bloques"] = df_dev.loc[
    df_dev['genero'] == 'main-captions', "nosw_score_milha_vw_cosin"]

df_dev.loc[df_dev['genero'] == 'main-forums', "best_bloques"] = df_dev.loc[
    df_dev['genero'] == 'main-forums', "nosw_score_aligned"]

corr_best_dev = df_dev[['num2', 'best_bloques']].corr().loc["best_bloques", "num2"]

df_test.loc[df_test['genero'] == 'main-news', "best_bloques"] = df_test.loc[
    df_test['genero'] == 'main-news', "nosw_score_aligned"]

df_test.loc[df_test['genero'] == 'main-captions', "best_bloques"] = df_test.loc[
    df_test['genero'] == 'main-captions', "nosw_score_milha_vw_cosin"]

df_test.loc[df_test['genero'] == 'main-forums', "best_bloques"] = df_test.loc[
    df_test['genero'] == 'main-forums', "nosw_score_aligned"]

corr_best_test = df_test[['num2', 'best_bloques']].corr().loc["best_bloques", "num2"]
"""
# Acercamiento 1 -> Un modelo para el dataset completo.

# Sin PCA

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('LR', LinearRegression())])

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

pipe.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]
pipe.score(x_dev, y_dev)
prob_train = pipe.predict(x_train)
prob_dev = pipe.predict(x_dev)
print("Regresion Lineal train", np.corrcoef(y_train, prob_train)[0, 1])
print("Regresion Lineal dev", np.corrcoef(y_dev, prob_dev)[0, 1])

# save the model to disk
filename = 'res/lr_model.sav'
pickle.dump(pipe, open(filename, 'wb'))

"""
# Con PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pca = PCA(n_components = 100)
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('PCA', pca),
                 ('LR', LinearRegression())])

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

pipe.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]
pipe.score(x_dev, y_dev)
prob_dev = pipe.predict(x_dev)
print("Regresion Lineal con PCA", np.corrcoef(y_dev, prob_dev)[0,1])
"""

# Random Forest
# Sin PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('RF', RandomForestRegressor())])

opt = BayesSearchCV(
    pipe,
    {
        'RF__n_estimators': Integer(10, 100),
        'RF__max_depth': Integer(1, 8)
    },
    n_iter=30,
    random_state=0)

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

opt.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]

prob_train = opt.predict(x_train)
prob_dev = opt.predict(x_dev)
print("Regresion RF", np.corrcoef(y_train, prob_train)[0, 1])
print("Regresion RF", np.corrcoef(y_dev, prob_dev)[0, 1])

# save the model to disk
filename = 'res/rf_model.sav'
pickle.dump(opt, open(filename, 'wb'))

"""
# Con PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pca = PCA(n_components = 30)
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('pca', pca),
                 ('RF', RandomForestRegressor())])

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

pipe.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]
pipe.score(x_dev, y_dev)
prob_dev = pipe.predict(x_dev)
print("Regresion RF con PCA", np.corrcoef(y_dev, prob_dev)[0,1])
"""
# LGBM
# Sin PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('lgbm', LGBMRegressor())])
opt = BayesSearchCV(
    pipe,
    {
        'lgbm__num_leaves': Integer(10, 100),
        'lgbm__learning_rate': Real(1e-3, 15, prior='log-uniform'),
        'lgbm__n_estimators': Integer(30, 300)
    },
    n_iter=30,
    random_state=0)

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

opt.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]
prob_train = opt.predict(x_train)
prob_dev = opt.predict(x_dev)
print("Regresion lgbm", np.corrcoef(y_train, prob_train)[0, 1])
print("Regresion lgbm", np.corrcoef(y_dev, prob_dev)[0, 1])

# save the model to disk
filename = 'res/lgbm_model.sav'
pickle.dump(opt, open(filename, 'wb'))

# SVR
# Sin PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]
pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('SVR', SVR())])
opt = BayesSearchCV(
    pipe,
    {
        'SVR__C': Real(1e-3, 1e+2, prior='log-uniform'),
        'SVR__degree': Integer(1, 8),
        'SVR__kernel': Categorical(['poly'])
    },
    n_iter=30,
    random_state=0)

opt.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]

prob_dev = opt.predict(x_dev)
prob_train = opt.predict(x_train)
print("Regresion SVR train", np.corrcoef(y_train, prob_train)[0, 1])
print("Regresion SVR dev", np.corrcoef(y_dev, prob_dev)[0, 1])

# save the model to disk
filename = 'res/SVR_model.sav'
pickle.dump(opt, open(filename, 'wb'))


# MLP
# Sin PCA
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

pipe = Pipeline([('imputer', imp),
                 ('scaler', StandardScaler()),
                 ('mlp', MLPRegressor())])

opt = GridSearchCV(estimator=pipe,
                   param_grid={'mlp__hidden_layer_sizes': [[100,], [100, 50, 5], [10, 50, 10], [75, 50, 25], [60, 20, 5]]})

y_train = df_train['num2']
x_train = df_train[df_train.columns[15:]]

opt.fit(x_train, y_train)

y_dev = df_dev['num2']
x_dev = df_dev[df_dev.columns[15:]]

prob_train = opt.predict(x_train)
prob_dev = opt.predict(x_dev)
print("Regresion MLP", np.corrcoef(y_train, prob_train)[0, 1])
print("Regresion MLP", np.corrcoef(y_dev, prob_dev)[0, 1])

# save the model to disk
filename = 'res/MLP_model.sav'
pickle.dump(opt, open(filename, 'wb'))