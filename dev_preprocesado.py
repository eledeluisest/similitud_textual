"""
Proyecto fin de asignatura NLP
Luis Esteban Andaluz

20/03/2021
dev_preprocesado.py

Preprocesado de conjuntos de entrenamiento para generacion de dataset sobre el que entrenar el modelo final.
Dependencias:
utilities.py
model = kv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True )

"""
from utilities import *
import time
import sys
print("importando datos...")
df_dev = f_tokeniza_y_estructura("corpus/stsbenchmark/sts-dev.csv")
i = int(sys.argv[1])
r_init = i * 375
r_fin = (i + 1) * 375
if r_fin > 1500:
    r_fin = 1500
df_dev = df_dev.iloc[r_init:r_fin]
print("1.- Datos Importados")
list_distancias = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']
list_distancias_wv = ['cosin', 'dot', 'dist']
list_corpus = ['wordnet_ic', 'brown_ic', 'semcor_ic']
list_umbrales = [0.5, 0.75, 0.85, 0.9]
palabras = [w for w in brown.words()]

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

####################
# CON STOPWORDS
####################
print("2. Proceso con StopWords")
df_dev.loc[:, 'score_aligned'] = df_dev. \
    apply(lambda x: f_devuelve_align(x['s1_tag'], x['s2_tag']), axis=1)
print("2.0 Score aligned Ok")
# Aqui podemos iterar en distancias[3+3], corpus [brown y el otro] y umbrales [0.5, 0.75, 0.85, 0.9] -> 60 scores
for dist in list_distancias:
    for corpus in list_corpus:
        for umbral in list_umbrales:
            print(dist, corpus, umbral, time.asctime())
            if corpus == 'wordnet_ic' and dist in ('path', 'lch', 'wup'):
                df_dev.loc[:, "_".join(['sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag'], tipo_dist=dist, corpus=corpus,
                                                         umbral=umbral), axis=1)
            else:
                print('No calculo')
            if corpus == 'brown_ic':
                df_dev.loc[:, "_".join(['sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag'], tipo_dist=dist, corpus=corpus,
                                                         corpus_ic=brown_ic, umbral=umbral), axis=1)
            if corpus == 'semcor_ic':
                df_dev.loc[:, "_".join(['sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag'], tipo_dist=dist, corpus=corpus,
                                                         corpus_ic=semcor_ic, umbral=umbral), axis=1)

print("2.1 Etapa aligned relajado Ok")
for dist in list_distancias:
    for corpus in list_corpus:
        print(dist, corpus, time.asctime())
        if corpus == 'wordnet_ic' and dist in ('path', 'lch', 'wup'):
            df_dev.loc[:, "_".join(['sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag'], x['s2_tag'], palabras_corpus=palabras, tipo_dist=dist,
                                              corpus=corpus), axis=1)
        else:
            print('No calculo')
        if corpus == 'brown_ic':
            df_dev.loc[:, "_".join(['sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag'], x['s2_tag'], palabras_corpus=palabras, tipo_dist=dist,
                                              corpus=corpus, corpus_ic=brown_ic), axis=1)
        if corpus == 'semcor_ic':
            df_dev.loc[:, "_".join(['sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag'], x['s2_tag'], palabras_corpus=palabras, tipo_dist=dist,
                                              corpus=corpus, corpus_ic=semcor_ic), axis=1)
print("2.2 Etapa mihalcea Ok")
model = kv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("2.3 Modelo Word2Vec importado")
# Aqui podemos iterar en distancias y umbrales -> 12 scores
for dist in list_distancias_wv:
    for umbral in list_umbrales:
        df_dev.loc[:, "_".join(['vw_rel', dist, str(umbral)])] = df_dev. \
            apply(lambda x: f_devuelve_align_rel_vw(x['s1_tag'], x['s2_tag'], model, tipo_sim=dist, umbral=umbral),
                  axis=1)
print("2.4 word2vec y aligned relajado ok")
# Aqui podemos iterar en distancias -> 3 scores
for dist in list_distancias_wv:
    df_dev.loc[:, 'score_milha_vw' + '_' + dist] = df_dev. \
        apply(lambda x: f_devuelve_mihalcea_vw(x['s1_tag'], x['s2_tag'], model, palabras, tipo_sim=dist), axis=1)
print("2.5 word2vec y mihalcea ok")
# 90 scores con stopwords.

####################
# SIN STOPWORDS
####################
print("3. Proceso sin StopWords")
df_dev.loc[:, 'nosw_score_aligned'] = df_dev. \
    apply(lambda x: f_devuelve_align(x['s1_tag_nosw'], x['s2_tag_nosw']), axis=1)
print("3.0 Score aligned Ok")
# Aqui podemos iterar en distancias[3+3], corpus [brown y el otro] y umbrales [0.5, 0.75, 0.85, 0.9] -> 60 scores
for dist in list_distancias:
    for corpus in list_corpus:
        for umbral in list_umbrales:
            if corpus == 'wordnet_ic' and dist in ('path', 'lch', 'wup'):
                df_dev.loc[:, "_".join(['nosw_sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(
                    lambda x: f_devuelve_align_rel(x['s1_tag_nosw'], x['s2_tag_nosw'], tipo_dist=dist, corpus=corpus,
                                                   umbral=umbral), axis=1)
            else:
                print('No calculo')
            if corpus == 'brown_ic':
                df_dev.loc[:, "_".join(['nosw_sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(
                    lambda x: f_devuelve_align_rel(x['s1_tag_nosw'], x['s2_tag_nosw'], tipo_dist=dist, corpus=corpus,
                                                   corpus_ic=brown_ic, umbral=umbral), axis=1)
            if corpus == 'semcor_ic':
                df_dev.loc[:, "_".join(['nosw_sem_rel', dist, corpus, str(umbral)])] = df_dev. \
                    apply(
                    lambda x: f_devuelve_align_rel(x['s1_tag_nosw'], x['s2_tag_nosw'], tipo_dist=dist, corpus=corpus,
                                                   corpus_ic=semcor_ic, umbral=umbral), axis=1)
print("3.1 Etapa aligned relajado Ok")
for dist in list_distancias:
    for corpus in list_corpus:
        if corpus == 'wordnet_ic' and dist in ('path', 'lch', 'wup'):
            df_dev.loc[:, "_".join(['nosw_sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag_nosw'], x['s2_tag_nosw'], palabras_corpus=palabras,
                                              tipo_dist=dist, corpus=corpus), axis=1)
        else:
            print('No calculo')
        if corpus == 'brown_ic':
            df_dev.loc[:, "_".join(['nosw_sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag_nosw'], x['s2_tag_nosw'], palabras_corpus=palabras,
                                              tipo_dist=dist, corpus=corpus, corpus_ic=brown_ic), axis=1)
        if corpus == 'semcor_ic':
            df_dev.loc[:, "_".join(['nosw_sem_mih', dist, corpus])] = df_dev. \
                apply(
                lambda x: f_devuelve_mihalcea(x['s1_tag_nosw'], x['s2_tag_nosw'], palabras_corpus=palabras,
                                              tipo_dist=dist, corpus=corpus, corpus_ic=semcor_ic), axis=1)
print("3.2 Etapa mihalcea Ok")
# Aqui podemos iterar en distancias y umbrales -> 12 scores
for dist in list_distancias_wv:
    for umbral in list_umbrales:
        df_dev.loc[:, "_".join(['nosw_vw_rel', dist, str(umbral)])] = df_dev. \
            apply(
            lambda x: f_devuelve_align_rel_vw(x['s1_tag_nosw'], x['s2_tag_nosw'], model, tipo_sim=dist, umbral=umbral),
            axis=1)
print("3.3 word2vec y aligned relajado ok")
# Aqui podemos iterar en distancias -> 3 scores
for dist in list_distancias_wv:
    df_dev.loc[:, 'nosw_score_milha_vw' + '_' + dist] = df_dev. \
        apply(lambda x: f_devuelve_mihalcea_vw(x['s1_tag_nosw'], x['s2_tag_nosw'], model, palabras, tipo_sim=dist),
              axis=1)
print("3.4 word2vec y mihalcea ok")
# 90 scores con stopwords.

# Total 180 scores.
print("Escribiendo resultados")
df_dev.to_csv('data/dev_preprocessed' + str(i) + '.csv', sep=';', index=False)
print("GoodBye")
