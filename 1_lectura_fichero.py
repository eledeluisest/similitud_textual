import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
import pandas as pd
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

df_train['s1_tag'] = df_train['s1_tokenized'].apply(pos_tag)
df_train['s2_tag'] = df_train['s2_tokenized'].apply(pos_tag)

pos_tag(df_train['s1_tokenized'].iloc[0])

wn.synsets(lemma=df_train['s1_tag'].iloc[0][0][0], pos=df_train['s1_tag'].iloc[0][0][1])
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