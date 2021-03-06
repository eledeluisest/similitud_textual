
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('wordnet_ic')
# nltk.download('brown')
from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.corpus import brown
from nltk.corpus import stopwords
from gensim.models import Word2Vec as wv, KeyedVectors as kv
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

with open("corpus/stsbenchmark/sts-train.csv", "r", encoding='utf-8') as f:
    texto = [x.replace('\n','').split('\t') for x in f.readlines()]

print("Total Lineas")
print(len(texto))
texto_7col = [t for t in texto if len(t) == 7]
print("Lineas Ok")
print(len(texto_7col))
texto_no7col = [t for t in texto if len(t) != 7]
print("Lineas No Ok")

texto_corregido = [x[:7] for x in texto_no7col]

print(len(texto_no7col))
texto_7col.extend(texto_corregido)
df_train = pd.DataFrame(texto_7col,
                        columns=['genero', 'dataset', 'ano', 'num1', 'num2', 's1', 's2'])
df_train['num2'] = df_train['num2'].astype(float)
for col in df_train.columns:
    print(col)
    print(df_train[col].value_counts())
    print("="*100)

tokenizer = RegexpTokenizer(r'\w+')
df_train['s1_tokenized'] = df_train['s1'].str.lower().apply(tokenizer.tokenize)
df_train['s2_tokenized'] = df_train['s2'].str.lower().apply(tokenizer.tokenize)

df_train['s1_tok_nosw'] = df_train['s1_tokenized'].apply(lambda x: [y for y in x if y not in stopwords.words("english")])
df_train['s2_tok_nosw'] = df_train['s2_tokenized'].apply(lambda x: [y for y in x if y not in stopwords.words("english")])

df_train['s1_tag'] = df_train['s1_tokenized'].apply(pos_tag)
df_train['s2_tag'] = df_train['s2_tokenized'].apply(pos_tag)


def f_devuelve_synset(palabra, pos):
    if pos[0].lower() in ('a', 'n', 'v'):
        return wn.synsets(lemma=palabra, pos=pos[0].lower())
    else:
        return wn.synsets(lemma=palabra)

semcor_ic = wordnet_ic.ic('ic-semcor.dat')
def f_devuelve_sim_syn(list_syn1, list_syn2, tipo_dist, corpus='brown_ic', verbose=False):
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
        brown_ic = wordnet_ic.ic('ic-brown.dat')
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
                    sim = syn1.res_similarity(syn2, brown_ic)
                elif tipo_dist == 'jcn':
                    sim = syn1.jcn_similarity(syn2, brown_ic)
                elif tipo_dist == 'lin':
                    sim = syn1.lin_similarity(syn2, brown_ic)
                if sim != None:
                    similitud.append(sim)
    elif corpus == 'semcor_ic':
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
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
                    sim = syn1.res_similarity(syn2, semcor_ic)
                elif tipo_dist == 'jcn':
                    sim = syn1.jcn_similarity(syn2, semcor_ic)
                elif tipo_dist == 'lin':
                    sim = syn1.lin_similarity(syn2, semcor_ic)
                if sim != None:
                    similitud.append(sim)
    if len(similitud) > 0:
        # print(similitud)
        return max(similitud)
    else:
        return 0

def f_devuelve_align(oracion1, oracion2, verbose=False):
    alineamiento = []
    for p1, t1 in oracion1:
        encontrado = False
        for p2, t2 in oracion2:
            if verbose:
                print(p1, p2)
            if p1 == p2 and not encontrado:
                alineamiento.append(1)
                encontrado = True
    return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))

def f_devuelve_align_rel(oracion1, oracion2, tipo_dist = 'path', umbral = 0.8, verbose=False):
    alineamiento = []
    primera_vuelta = True
    dic_sin2 = {}
    for p1, t1 in oracion1:
        encontrado = False
        sin1 = f_devuelve_synset(p1, t1)
        for p2, t2 in oracion2:
            if primera_vuelta:
                dic_sin2[p2] = f_devuelve_synset(p2, t2)
            sim = f_devuelve_sim_syn(sin1, dic_sin2[p2], tipo_dist)
            if verbose:
                print(p1, p2)
            if sim > umbral and not encontrado:
                alineamiento.append(1)
                encontrado = True
        primera_vuelta = False
    return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))

def f_idf(palabra, palabras):
    frec = len([x for x in palabras if palabra == x])
    if frec > 0:
        return np.log(len(palabras) /frec)
    else:
        return 0

def f_devuelve_mihalcea(oracion1, oracion2,palabras_corpus, tipo_dist = 'path', verbose=False):
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
            similitudes[(p1, p2)] = f_devuelve_sim_syn(sin1, sinsets2[p2], tipo_dist)
        primera_vuelta = False
    den1 = sum(dic_idf_or1.values())
    den2 = sum(dic_idf_or2.values())
    num1 = 0
    for p1, t1 in oracion1:
        posibilidad = []
        for k in similitudes.keys():
            if k[0] == p1:
                posibilidad.append(similitudes[k])
        num1 += dic_idf_or1[p1] * max(posibilidad)
    num2 = 0
    for p2, t2 in oracion2:
        posibilidad = []
        for k in similitudes.keys():
            if k[1] == p2:
                posibilidad.append(similitudes[k])
        num2 += dic_idf_or2[p2] * max(posibilidad)
    return 0.5*(num1/den1 + num2/den2)



df_train.loc[:10,'score_aligned'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_align(x['s1_tag'], x['s2_tag']), axis=1)

df_train.loc[:10,'score_rel'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag']), axis=1)

df_train.loc[:10,'score_rel_lch'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag'], tipo_dist = 'lch'), axis=1)

df_train.loc[:10,'score_rel_wup'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_align_rel(x['s1_tag'], x['s2_tag'], tipo_dist = 'wup'), axis=1)

palabras = [w for w in brown.words()]
df_train.loc[:10,'score_milh'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_mihalcea(x['s1_tag'], x['s2_tag'], palabras_corpus=palabras), axis=1)


model = kv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True )

def f_sim_cosin_vw(w1, w2, model):
    try:
        return model.similarity(w1, w2)
    except KeyError:
        return 0

def f_sim_prod_vw(w1, w2, model):
    try:
        return np.dot(model[w1], model[w2])
    except KeyError:
        return 0

def f_sim_dist_vw(w1, w2, model):
    try:
        return np.linalg.norm(model[w1] - model[w2])
    except KeyError:
        return 0


def f_devuelve_align_rel_vw(oracion1, oracion2, tipo_sim = 'cosin', umbral = 0.8, model=model, verbose=False):
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
    return 2*sum(alineamiento)/(len(oracion1)+len(oracion2))


df_train.loc[:10,'score_alig_vw'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_align_rel_vw(x['s1_tag'], x['s2_tag']), axis=1)

def f_devuelve_mihalcea_vw(oracion1, oracion2,palabras_corpus, tipo_sim = 'cosin', verbose=False):
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
        posibilidad = []
        for k in similitudes.keys():
            if k[0] == p1:
                posibilidad.append(similitudes[k])
        num1 += dic_idf_or1[p1] * max(posibilidad)
    num2 = 0
    for p2, t2 in oracion2:
        posibilidad = []
        for k in similitudes.keys():
            if k[1] == p2:
                posibilidad.append(similitudes[k])
        num2 += dic_idf_or2[p2] * max(posibilidad)
    return 0.5*(num1/den1 + num2/den2)


df_train.loc[:10,'score_milha_vw'] = df_train.iloc[:10].\
    apply(lambda x: f_devuelve_mihalcea_vw(x['s1_tag'], x['s2_tag'], palabras), axis=1)



df_train.iloc[:10,:]

wv.load("GoogleNews-vectors-negative300.bin.gz")




"""
genero
main-news        3299
main-captions    2000
main-forum        450
Name: genero, dtype: int64
====================================================================================================
dataset
headlines     1999
MSRvid        1000
images        1000
MSRpar        1000
deft-forum     450
deft-news      300
Name: dataset, dtype: int64
====================================================================================================
ano
2014         1856
2015         1099
2012train    1004
2012test      996
2013          597
2016          197
Name: ano, dtype: int64
====================================================================================================
num1
0081    11
0217    11
0192    11
0158    11
0240    11
        ..
0970     1
1345     1
1357     1
1151     1
0859     1
Name: num1, Length: 1244, dtype: int64
====================================================================================================
num2
0.000    367
4.000    354
3.000    315
3.800    267
5.000    266
        ... 
3.769      1
1.273      1
3.692      1
3.765      1
0.643      1
Name: num2, Length: 140, dtype: int64
====================================================================================================
"""