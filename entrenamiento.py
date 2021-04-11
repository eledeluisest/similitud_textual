# Entrenamiento de un modelo de ML
import pandas as pd
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

df_train = pd.read_csv("data/train_features.csv", sep=';')
df_dev = pd.read_csv("data/dev_features.csv", sep=';')
df_test = pd.read_csv("data/test_features.csv", sep=';')
# Base line -> Mejor métrica y combinación de mejores métricas para cada bloque
# Mejor metrica
corr_baseline_train = df_train[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_dev = df_dev[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]
corr_baseline_test = df_test[['num2', 'nosw_score_aligned']].corr().loc["nosw_score_aligned", "num2"]

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


# Acercamiento 2 -> Un modelo para cada bloque y combinación
