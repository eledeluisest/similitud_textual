"""
Proyecto fin de asignatura NLP
Luis Esteban Andaluz

20/03/2021
utilities.py

Funciones necesarias para el desarrollo del proyecto
Dependencias:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('wordnet_ic')
# nltk.download('brown')

"""
import nltk


from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.corpus import brown
from nltk.corpus import stopwords
from gensim.models import Word2Vec as wv, KeyedVectors as kv
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np

print("loading utilities...")
def f_tokeniza_y_estructura(file, nrow=-1):
    """
    Lee fichero y lo estructura en un dataframe de pandas. Despues aplica tecnicas de tokenizado y eliminacion de stopwords
    :param file: fichero de lectura
    :return: dataframe de pandas con la informacion
    """
    with open(file, "r", encoding='utf-8') as f:
        texto = [x.replace('\n', '').split('\t') for x in f.readlines()]
    texto_7col = [t for t in texto if len(t) == 7]
    texto_no7col = [t for t in texto if len(t) != 7]
    texto_corregido = [x[:7] for x in texto_no7col]
    texto_7col.extend(texto_corregido)
    if nrow > 0:
        df_train = pd.DataFrame(texto_7col[:nrow],
                                columns=['genero', 'dataset', 'ano', 'num1', 'num2', 's1', 's2'])
    else:
        df_train = pd.DataFrame(texto_7col,
                                columns=['genero', 'dataset', 'ano', 'num1', 'num2', 's1', 's2'])
    df_train['num2'] = df_train['num2'].astype(float)

    tokenizer = RegexpTokenizer(r'\w+')
    df_train['s1_tokenized'] = df_train['s1'].str.lower().apply(tokenizer.tokenize)
    df_train['s2_tokenized'] = df_train['s2'].str.lower().apply(tokenizer.tokenize)

    df_train['s1_tok_nosw'] = df_train['s1_tokenized'].apply(
        lambda x: [y for y in x if y not in stopwords.words("english")])
    df_train['s2_tok_nosw'] = df_train['s2_tokenized'].apply(
        lambda x: [y for y in x if y not in stopwords.words("english")])

    df_train['s1_tag'] = df_train['s1_tokenized'].apply(pos_tag)
    df_train['s2_tag'] = df_train['s2_tokenized'].apply(pos_tag)

    df_train['s1_tag_nosw'] = df_train['s1_tok_nosw'].apply(pos_tag)
    df_train['s2_tag_nosw'] = df_train['s2_tok_nosw'].apply(pos_tag)

    return df_train


def f_devuelve_synset(palabra, pos):
    """
    Calculo del synset asociado a una palabra y etiqueta
    :param palabra: palabra para encontrar el synset
    :param pos: etiqueta morfologica
    :return: synset de la palabra / etiqueta (lista)
    """
    if pos[0].lower() in ('a', 'n', 'v'):
        return wn.synsets(lemma=palabra, pos=pos[0].lower())
    else:
        return wn.synsets(lemma=palabra)


def f_devuelve_sim_syn(list_syn1, list_syn2, tipo_dist, corpus='wordnet_ic', corpus_ic = None, verbose=False):
    """
    Calculo de la similitud entre synsets como el maximo de las similitudes entre todos los miembros de ambos synsets.
    :param list_syn1: Lista de synsets correspondientes a la primera palabra
    :param list_syn2:  Lista de synsets correspondientes a la segunda palabra
    :param tipo_dist: Tipo de similitud que se quiere calcular: path, lch, wup, res, jcn o lin. Las tres ultimas solo se pueden calcular con lso corpus brown_ic y semcor_ic
    :param corpus: corpus para el calculo de la similitud wordnet_ic, brown_ic o semcor_ic
    :param verbose:
    :return: Similitud entre palabras (float)
    """
    similitud = []
    if corpus == 'wordnet_ic':
        for syn1 in list_syn1:
            for syn2 in list_syn2:
                if verbose:
                    print(syn1, syn2, syn1.path_similarity(syn2))
                if tipo_dist == 'path':
                    sim = syn1.path_similarity(syn2)
                elif tipo_dist == 'lch':
                    if syn1.pos() == syn2.pos():
                        sim = syn1.lch_similarity(syn2)
                    else:
                        sim = 0
                elif tipo_dist == 'wup':
                    sim = syn1.wup_similarity(syn2)
                elif tipo_dist == 'res':
                    raise ValueError('No podemos calucular esta similitud con este corpus. Por favor, prueba con brown o semcor.')
                elif tipo_dist == 'jcn':
                    raise ValueError('No podemos calucular esta similitud con este corpus. Por favor, prueba con brown o semcor.')
                elif tipo_dist == 'lin':
                    raise ValueError('No podemos calucular esta similitud con este corpus. Por favor, prueba con brown o semcor.')
                if sim != None:
                    similitud.append(sim)
    elif corpus == 'brown_ic':
        brown_ic = corpus_ic
        for syn1 in list_syn1:
            for syn2 in list_syn2:
                if verbose:
                    print(syn1, syn2, syn1.path_similarity(syn2, brown_ic))
                if tipo_dist == 'path':
                    sim = syn1.path_similarity(syn2, brown_ic)
                elif tipo_dist == 'lch':
                    if syn1.pos() == syn2.pos():
                        sim = syn1.lch_similarity(syn2, brown_ic)
                    else:
                        sim = 0
                elif tipo_dist == 'wup':
                    sim = syn1.wup_similarity(syn2, brown_ic)
                elif tipo_dist == 'res':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.res_similarity(syn2, brown_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                elif tipo_dist == 'jcn':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.jcn_similarity(syn2, brown_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                elif tipo_dist == 'lin':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.lin_similarity(syn2, brown_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                if sim != None:
                    similitud.append(sim)
    elif corpus == 'semcor_ic':
        semcor_ic = corpus_ic
        for syn1 in list_syn1:
            for syn2 in list_syn2:
                if verbose:
                    print(syn1, syn2, syn1.path_similarity(syn2, semcor_ic))
                if tipo_dist == 'path':
                    sim = syn1.path_similarity(syn2, semcor_ic)
                elif tipo_dist == 'lch':
                    if syn1.pos() == syn2.pos():
                        sim = syn1.lch_similarity(syn2, semcor_ic)
                    else:
                        sim = 0
                elif tipo_dist == 'wup':
                    sim = syn1.wup_similarity(syn2, semcor_ic)
                elif tipo_dist == 'res':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.res_similarity(syn2, semcor_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                elif tipo_dist == 'jcn':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.jcn_similarity(syn2, semcor_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                elif tipo_dist == 'lin':
                    if syn1.pos() == syn2.pos():
                        try:
                            sim = syn1.lin_similarity(syn2, semcor_ic)
                        except Exception as e:
                            sim = 0
                    else:
                        sim = 0
                if sim != None:
                    similitud.append(sim)
    if len(similitud) > 0:
        # print(similitud)
        return max(similitud)
    else:
        return 0

def f_devuelve_align(oracion1, oracion2, verbose=False):
    """
    Calculo del alineamiento
    :param oracion1: lista de palabras y etiquetas de la primera oracion
    :param oracion2: lista de palabras y etiquetas de la segunda oracion
    :param verbose:
    :return: metrica de alineamiento (float)
    """
    alineamiento = []
    for p1, t1 in oracion1:
        encontrado = False
        for p2, t2 in oracion2:
            if verbose:
                print(p1, p2)
            if p1 == p2 and not encontrado:
                alineamiento.append(1)
                encontrado = True
    if len(oracion1) > 0 and len(oracion2) > 0:
        return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))
    else:
        return 0

def f_devuelve_align_rel(oracion1, oracion2, tipo_dist = 'path', corpus='wordnet_ic', corpus_ic = None, umbral = 0.8, verbose=False):
    """
    Calculo relajado del alineamiento teniendo en cuenta similitud semantica entre palabras y umbrales para medir el acierto.
    :param oracion1: lista de palabras y etiquetas de la primera oracion
    :param oracion2: lista de palabras y etiquetas de la segunda oracion
    :param tipo_dist: forma de medir la similitud entre synsets (mismos que en f_devuelve_sim_syn)
    :param corpus: corpus para medicion de similitud
    :param umbral: umbral para considerar el aciert
    :param verbose:
    :return: metrica de alineamiento relajado (float)
    """
    alineamiento = []
    primera_vuelta = True
    dic_sin2 = {}
    for p1, t1 in oracion1:
        encontrado = False
        sin1 = f_devuelve_synset(p1, t1)
        for p2, t2 in oracion2:
            if primera_vuelta:
                dic_sin2[p2] = f_devuelve_synset(p2, t2)
            sim = f_devuelve_sim_syn(sin1, dic_sin2[p2], tipo_dist, corpus, corpus_ic)
            if verbose:
                print(p1, p2)
            if sim > umbral and not encontrado:
                alineamiento.append(1)
                encontrado = True
        primera_vuelta = False
    if len(oracion1) > 0 and len(oracion2) > 0:
        return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))
    else:
        return 0

def f_idf(palabra, palabras):
    """
    Calculo de la idf de una palabra a partir de las palabras de algun corpus
    :param palabra: palabra objetivo del calculo
    :param palabras: paabras de corpus de referencia
    :return: métrica de idf (float)
    """
    frec = len([x for x in palabras if palabra == x])
    if frec > 0:
        return np.log(len(palabras) /frec)
    else:
        return 0

def f_devuelve_mihalcea(oracion1, oracion2,palabras_corpus, tipo_dist = 'path', corpus='wordnet_ic', corpus_ic = None, verbose=False):
    """
    Calculo de la metrica de similitud de milhacea.
    :param corpus_ic:
    :param oracion1: Lista con palabras y etiquetas de la primera oracion
    :param oracion2: Lista con palabras y etiquetas de la segunda oracon
    :param palabras_corpus:  Lista con palabras de algun corpus para el calculo del idf
    :param tipo_dist: Forma de calcular la similitud semantica entre palabras
    :param verbose:
    :return: metrica de milhacea (float)
    """
    dic_idf_or1 = {}
    dic_idf_or2 = {}
    primera_vuelta = True
    sinsets2 = {}
    similitudes = {}
    for p1, t1 in oracion1:
        sin1 = f_devuelve_synset(p1, t1)
        idf1 = f_idf(p1, palabras_corpus)
        dic_idf_or1[p1] = idf1
        for p2, t2 in oracion2:
            if primera_vuelta:
                sin2 = f_devuelve_synset(p2, t2)
                sinsets2[p2] = sin2
                idf2 = f_idf(p2, palabras_corpus)
                dic_idf_or2[p2] = idf2
            similitudes[(p1, p2)] = f_devuelve_sim_syn(sin1, sinsets2[p2], tipo_dist, corpus, corpus_ic)
        primera_vuelta = False
    den1 = sum(dic_idf_or1.values())
    den2 = sum(dic_idf_or2.values())
    num1 = 0
    for p1, t1 in oracion1:
        posibilidad = [0]
        for k in similitudes.keys():
            if k[0] == p1:
                posibilidad.append(similitudes[k])
        try:
            num1 += dic_idf_or1[p1] * max(posibilidad)
        except Exception as e:
            print(e)
            print(oracion1)
    num2 = 0
    for p2, t2 in oracion2:
        posibilidad = [0]
        for k in similitudes.keys():
            if k[1] == p2:
                posibilidad.append(similitudes[k])
        try:
            num2 += dic_idf_or2[p2] * max(posibilidad)
        except Exception as e:
            print(e)
            print(oracion2)
    if den1 == 0 or den2 == 0:
        return 0
    else:
        return 0.5*(num1/den1 + num2/den2)

def f_sim_cosin_vw(w1, w2, model):
    """
    Calculo de similitud coseno segun algun modelo  distribuido
    :param w1: Primera palabra
    :param w2: Segunda palabra
    :param model: Modelo distribuido (gensim.models)
    :return: metrica de similitud (float)
    """
    try:
        return model.similarity(w1, w2)
    except KeyError:
        return 0

def f_sim_prod_vw(w1, w2, model):
    """
    Calculo de la similitud producto de algun modelo distribuido
    :param w1: Primera palabra
    :param w2: Segunda palabra
    :param model: Modelo distribuido (gensim.models)
    :return:  metrica de similitud (float)
    """
    try:
        return np.dot(model[w1], model[w2])
    except KeyError:
        return 0

def f_sim_dist_vw(w1, w2, model):
    """
    Calculo de la similitud distancia de algun modelo distribuido
    :param w1: Primera palabra
    :param w2: Segunda palabra
    :param model: Modelo distribuido (gensim.models)
    :return:  metrica de similitud (float)
    """
    try:
        return np.linalg.norm(model[w1] - model[w2])
    except KeyError:
        return 0


def f_devuelve_align_rel_vw(oracion1, oracion2, model, tipo_sim = 'cosin', umbral = 0.8, verbose=False):
    """
    Calculo relajado del alineamiento teniendo en cuenta similitud distribuida entre palabras y umbrales para medir el acierto.
    :param oracion1: lista de palabras y etiquetas de la primera oracion
    :param oracion2: lista de palabras y etiquetas de la segunda oracion
    :param tipo_dist: forma de medir la similitud entre palabras: cosin, dot o dist.
    :param model: modelo
    :param umbral: umbral para considerar el aciert
    :param verbose:
    :return: metrica de alineamiento relajado (float)
    """
    alineamiento = []
    for p1, t1 in oracion1:
        encontrado = False
        for p2, t2 in oracion2:
            if tipo_sim == 'cosin':
                sim = f_sim_cosin_vw(p1, p2, model=model)
            if tipo_sim == 'dot':
                sim = f_sim_prod_vw(p1, p2, model=model)
            if tipo_sim == 'dist':
                sim = f_sim_dist_vw(p1, p2, model=model)
            if verbose:
                print(p1, p2)
            if sim > umbral and not encontrado:
                alineamiento.append(1)
                encontrado = True
    if len(oracion1) > 0 and len(oracion2) > 0:
        return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))
    else:
        return 0

def f_devuelve_mihalcea_vw(oracion1, oracion2, model, palabras_corpus, tipo_sim = 'cosin'):
    """
    Calculo de la metrica de similitud de milhacea.
    :param oracion1: Lista con palabras y etiquetas de la primera oracion
    :param oracion2: Lista con palabras y etiquetas de la segunda oracon
    :param palabras_corpus:  Lista con palabras de algun corpus para el calculo del idf
    :param tipo_dist: forma de medir la similitud entre palabras: cosin, dot o dist.
    :param tipo_dist: forma de medir la similitud entre palabras: cosin, dot o dist.
    :param verbose:
    :return: metrica de milhacea (float)
    """
    dic_idf_or1 = {}
    dic_idf_or2 = {}
    primera_vuelta = True
    sinsets2 = {}
    similitudes = {}
    for p1, t1 in oracion1:
        idf1 = f_idf(p1, palabras_corpus)
        dic_idf_or1[p1] = idf1
        for p2, t2 in oracion2:
            if primera_vuelta:
                sin2 = f_devuelve_synset(p2, t2)
                sinsets2[p2] = sin2
                idf2 = f_idf(p2, palabras_corpus)
                dic_idf_or2[p2] = idf2
            if tipo_sim == 'cosin':
                similitudes[(p1, p2)] = f_sim_cosin_vw(p1, p2, model=model)
            if tipo_sim == 'dot':
                similitudes[(p1, p2)] = f_sim_prod_vw(p1, p2, model=model)
            if tipo_sim == 'dist':
                similitudes[(p1, p2)] = f_sim_dist_vw(p1, p2, model=model)
        primera_vuelta = False
    den1 = sum(dic_idf_or1.values())
    den2 = sum(dic_idf_or2.values())
    num1 = 0
    for p1, t1 in oracion1:
        posibilidad = [0]
        for k in similitudes.keys():
            if k[0] == p1:
                try:
                    posibilidad.append(similitudes[k])
                except Exception as e:
                    print(e)
                    print(oracion1)
        try:
            num1 += dic_idf_or1[p1] * max(posibilidad)
        except Exception as e:
            print(e)
            print(oracion1)
    num2 = 0
    for p2, t2 in oracion2:
        posibilidad = [0]
        for k in similitudes.keys():
            if k[1] == p2:
                try:
                    posibilidad.append(similitudes[k])
                except Exception as e:
                    print(e)
                    print(oracion2)
        try:
            num2 += dic_idf_or2[p2] * max(posibilidad)
        except Exception as e:
            print(e)
            print(oracion2)
    if den1 == 0 or den2 == 0:
        return 0
    else:
        return 0.5*(num1/den1 + num2/den2)



