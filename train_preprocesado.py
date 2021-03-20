"""
Proyecto fin de asignatura NLP
Luis Esteban Andaluz

20/03/2021
train_preprocesado.py

Preprocesado de conjuntos de entrenamiento para generacion de dataset sobre el que entrenar el modelo final.
Dependencias:
utilities.py
model = kv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True )

"""
from utilities import *

df_train = f_tokeniza_y_estructura("corpus/stsbenchmark/sts-train.csv")

