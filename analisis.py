import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df_train = pd.read_csv("data/train_features.csv", sep=';')
df_dev = pd.read_csv("data/dev_features.csv", sep=';')

# Estudio de características


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_train['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_train['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_train['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_train['len_or1'].median(),2))),
                   ]
df_train['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_train['len_or1_nosw'].hist(bins=15, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("train - all")
plt.savefig("res/train_all_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_dev['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_dev['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_dev['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_dev['len_or1'].median(),2))),
                   ]
df_dev['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_dev['len_or1_nosw'].hist(bins=15, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("dev - all")
plt.savefig("res/dev_all_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-news']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-news']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-news']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-news']['len_or1'].median(),2))),
                   ]
df_train[df_train['genero'] == 'main-news']['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_train[df_train['genero'] == 'main-news']['len_or1_nosw'].hist(bins=15, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("train - news")
plt.savefig("res/train_news_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-news']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-news']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-news']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-news']['len_or1'].median(),2))),
                   ]
df_dev[df_dev['genero'] == 'main-news']['len_or1'].hist(bins=13, density=True, alpha=0.6, color="red")
df_dev[df_dev['genero'] == 'main-news']['len_or1_nosw'].hist(bins=15, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("dev - news")
plt.savefig("res/dev_news_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-captions']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-captions']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-captions']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-captions']['len_or1'].median(),2))),
                   ]
df_train[df_train['genero'] == 'main-captions']['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_train[df_train['genero'] == 'main-captions']['len_or1_nosw'].hist(bins=13, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("train - captions")
plt.savefig("res/train_captions_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-captions']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-captions']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-captions']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-captions']['len_or1'].median(),2))),
                   ]
df_dev[df_dev['genero'] == 'main-captions']['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_dev[df_dev['genero'] == 'main-captions']['len_or1_nosw'].hist(bins=11, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("dev - captions")
plt.savefig("res/dev_captions_lon.png", bbox_inches="tight")
plt.show()
plt.clf()



legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-forum']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-forum']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_train[df_train['genero'] == 'main-forum']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_train[df_train['genero'] == 'main-forum']['len_or1'].median(),2))),
                   ]
df_train[df_train['genero'] == 'main-forum']['len_or1'].hist(bins=15, density=True, alpha=0.6, color="red")
df_train[df_train['genero'] == 'main-forum']['len_or1_nosw'].hist(bins=9, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("train - forum")
plt.savefig("res/train_forum_lon.png", bbox_inches="tight")
plt.show()
plt.clf()


legend_elements = [Line2D([0], [0], color='blue', lw=4, label="Sin stop words"),
                   Line2D([0], [0], color='blue', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-forums']['len_or1_nosw'].mean(),2))),
                   Line2D([0], [0], color='blue', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-forums']['len_or1_nosw'].median(),2))),
                   Line2D([0], [0], color='red', lw=4, label="Con stop words"),
                   Line2D([0], [0], color='red', lw=4, label='Media:'+str(round(df_dev[df_dev['genero'] == 'main-forums']['len_or1'].mean(),2))),
                   Line2D([0], [0], color='red', lw=4, label='Mediana:'+str(round(df_dev[df_dev['genero'] == 'main-forums']['len_or1'].median(),2))),
                   ]
df_dev[df_dev['genero'] == 'main-forums']['len_or1'].hist(bins=8, density=True, alpha=0.6, color="red")
df_dev[df_dev['genero'] == 'main-forums']['len_or1_nosw'].hist(bins=8, density=True, alpha=0.6, color="blue")
plt.legend(handles=legend_elements)
plt.title("dev - forum")
plt.savefig("res/dev_forum_lon.png", bbox_inches="tight")
plt.show()
plt.clf()

##############################
# Estudio de la correlación
##############################
# general
res_train = df_train.corr()['num2'].sort_values()
res_dev = df_dev.corr()['num2'].sort_values()

res_train.to_csv("res/top_train.csv", decimal=',', sep=';')
res_dev.to_csv("res/top_dev.csv", decimal=',', sep=';')

# genero == 'main-news'
res_train_news = df_train[df_train['genero'] == 'main-news'].corr()['num2'].sort_values()
res_dev_news = df_dev[df_dev['genero'] == 'main-news'].corr()['num2'].sort_values()

res_train_news.to_csv("res/top_train_news.csv", decimal=',', sep=';')
res_dev_news.to_csv("res/top_dev_news.csv", decimal=',', sep=';')

# genero == 'main-captions'
res_train_captions = df_train[df_train['genero'] == 'main-captions'].corr()['num2'].sort_values()
res_dev_captions = df_dev[df_dev['genero'] == 'main-captions'].corr()['num2'].sort_values()

res_train_captions.to_csv("res/top_train_captions.csv", decimal=',', sep=';')
res_dev_captions.to_csv("res/top_dev_captions.csv", decimal=',', sep=';')

# genero == 'main-forum'
res_train_forum = df_train[df_train['genero'] == 'main-forum'].corr()['num2'].sort_values()
res_dev_forum = df_dev[df_dev['genero'] == 'main-forums'].corr()['num2'].sort_values()

res_train_forum.to_csv("res/top_train_forum.csv", decimal=',', sep=';')
res_dev_forum.to_csv("res/top_dev_forum.csv", decimal=',', sep=';')


legend_elements = [Line2D([0], [0], color='blue', lw=4, label='vw'),
                   Line2D([0], [0], color='red', lw=4, label='semcor'),
                   Line2D([0], [0], color='orange', lw=4, label='brown'),
                   Line2D([0], [0], color='black', lw=4, label='wordnet'),
                   Line2D([0], [0], color='green', lw=4, label='aligned'),

                   Line2D([0], [0], marker='x', markersize=7, label='Aligned Relajado',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='o', markersize=7, label='Aligned',
                          markerfacecolor='g'),
                   Line2D([0], [0], marker='^', markersize=7, label='Mihalcea',
                          markerfacecolor='b')
                   ]

# stop words vs no stop words
# para general
df_res_train = res_train.reset_index()
df_res_train_consw = df_res_train[~df_res_train['index'].str.contains('nosw_')]
df_res_train_consw.loc[:, 'clave'] = df_res_train_consw['index']
df_res_train_consw.loc[:, 'corr_consw'] = df_res_train_consw['num2']
df_res_train_nosw = df_res_train[df_res_train['index'].str.contains('nosw_')]
df_res_train_nosw.loc[:, 'clave'] = df_res_train_nosw['index'].str[5:]
df_res_train_nosw.loc[:, 'corr_nosw'] = df_res_train_nosw['num2']
df_train_comp = pd.merge(df_res_train_consw[['clave', 'corr_consw']], df_res_train_nosw[['clave', 'corr_nosw']],
                         on=['clave'], how='inner')

df_train_comp.loc[df_train_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_train_comp.loc[df_train_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_train_comp.loc[df_train_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_train_comp.loc[df_train_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_train_comp.loc[df_train_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'
plt.rcParams["figure.figsize"] = (7.4, 6.4)
fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_train_comp.loc[df_train_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_train_comp.loc[~df_train_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("train - all")
plt.xlim([-0.25, 0.75])
plt.ylim([-0.25, 0.75])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/train_all.png", bbox_inches="tight")
plt.show()

df_res_dev = res_dev.reset_index()
df_res_dev_consw = df_res_dev[~df_res_dev['index'].str.contains('nosw_')]
df_res_dev_consw.loc[:, 'clave'] = df_res_dev_consw['index']
df_res_dev_consw.loc[:, 'corr_consw'] = df_res_dev_consw['num2']
df_res_dev_nosw = df_res_dev[df_res_dev['index'].str.contains('nosw_')]
df_res_dev_nosw.loc[:, 'clave'] = df_res_dev_nosw['index'].str[5:]
df_res_dev_nosw.loc[:, 'corr_nosw'] = df_res_dev_nosw['num2']
df_dev_comp = pd.merge(df_res_dev_consw[['clave', 'corr_consw']], df_res_dev_nosw[['clave', 'corr_nosw']],
                       on=['clave'], how='inner')

df_dev_comp.loc[df_dev_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_dev_comp.loc[df_dev_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_dev_comp.loc[df_dev_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_dev_comp.loc[df_dev_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_dev_comp.loc[df_dev_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_dev_comp.loc[df_dev_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_dev_comp.loc[~df_dev_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("dev - all")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/dev_all.png", bbox_inches="tight")
plt.show()

# para genero news
df_res_train_news = res_train_news.reset_index()
df_res_train_news_consw = df_res_train_news[~df_res_train_news['index'].str.contains('nosw_')]
df_res_train_news_consw.loc[:, 'clave'] = df_res_train_news_consw['index']
df_res_train_news_consw.loc[:, 'corr_consw'] = df_res_train_news_consw['num2']
df_res_train_news_nosw = df_res_train_news[df_res_train_news['index'].str.contains('nosw_')]
df_res_train_news_nosw.loc[:, 'clave'] = df_res_train_news_nosw['index'].str[5:]
df_res_train_news_nosw.loc[:, 'corr_nosw'] = df_res_train_news_nosw['num2']
df_train_news_comp = pd.merge(df_res_train_news_consw[['clave', 'corr_consw']],
                              df_res_train_news_nosw[['clave', 'corr_nosw']],
                              on=['clave'], how='inner')

df_train_news_comp.loc[df_train_news_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_train_news_comp.loc[df_train_news_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_train_news_comp.loc[df_train_news_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_train_news_comp.loc[df_train_news_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_train_news_comp.loc[df_train_news_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_train_news_comp.loc[
        df_train_news_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_train_news_comp.loc[
    ~df_train_news_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("train - news")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/train_news.png", bbox_inches="tight")
plt.show()

df_res_dev_news = res_dev_news.reset_index()
df_res_dev_news_consw = df_res_dev_news[~df_res_dev_news['index'].str.contains('nosw_')]
df_res_dev_news_consw.loc[:, 'clave'] = df_res_dev_news_consw['index']
df_res_dev_news_consw.loc[:, 'corr_consw'] = df_res_dev_news_consw['num2']
df_res_dev_news_nosw = df_res_dev_news[df_res_dev_news['index'].str.contains('nosw_')]
df_res_dev_news_nosw.loc[:, 'clave'] = df_res_dev_news_nosw['index'].str[5:]
df_res_dev_news_nosw.loc[:, 'corr_nosw'] = df_res_dev_news_nosw['num2']
df_dev_news_comp = pd.merge(df_res_dev_news_consw[['clave', 'corr_consw']],
                            df_res_dev_news_nosw[['clave', 'corr_nosw']],
                            on=['clave'], how='inner')

df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_dev_news_comp.loc[df_dev_news_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_dev_news_comp.loc[
    ~df_dev_news_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("dev - news")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/dev_news.png", bbox_inches="tight")
plt.show()

# para genero captions
df_res_train_captions = res_train_captions.reset_index()
df_res_train_captions_consw = df_res_train_captions[~df_res_train_captions['index'].str.contains('nosw_')]
df_res_train_captions_consw.loc[:, 'clave'] = df_res_train_captions_consw['index']
df_res_train_captions_consw.loc[:, 'corr_consw'] = df_res_train_captions_consw['num2']
df_res_train_captions_nosw = df_res_train_captions[df_res_train_captions['index'].str.contains('nosw_')]
df_res_train_captions_nosw.loc[:, 'clave'] = df_res_train_captions_nosw['index'].str[5:]
df_res_train_captions_nosw.loc[:, 'corr_nosw'] = df_res_train_captions_nosw['num2']
df_train_captions_comp = pd.merge(df_res_train_captions_consw[['clave', 'corr_consw']],
                                  df_res_train_captions_nosw[['clave', 'corr_nosw']],
                                  on=['clave'], how='inner')

df_train_captions_comp.loc[df_train_captions_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_train_captions_comp.loc[df_train_captions_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_train_captions_comp.loc[df_train_captions_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_train_captions_comp.loc[df_train_captions_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_train_captions_comp.loc[df_train_captions_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_train_captions_comp.loc[
        df_train_captions_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_train_captions_comp.loc[
    ~df_train_captions_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("train - captions")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/train_captions.png", bbox_inches="tight")
plt.show()

df_res_dev_captions = res_dev_captions.reset_index()
df_res_dev_captions_consw = df_res_dev_captions[~df_res_dev_captions['index'].str.contains('nosw_')]
df_res_dev_captions_consw.loc[:, 'clave'] = df_res_dev_captions_consw['index']
df_res_dev_captions_consw.loc[:, 'corr_consw'] = df_res_dev_captions_consw['num2']
df_res_dev_captions_nosw = df_res_dev_captions[df_res_dev_captions['index'].str.contains('nosw_')]
df_res_dev_captions_nosw.loc[:, 'clave'] = df_res_dev_captions_nosw['index'].str[5:]
df_res_dev_captions_nosw.loc[:, 'corr_nosw'] = df_res_dev_captions_nosw['num2']
df_dev_captions_comp = pd.merge(df_res_dev_captions_consw[['clave', 'corr_consw']],
                                df_res_dev_captions_nosw[['clave', 'corr_nosw']],
                                on=['clave'], how='inner')

df_dev_captions_comp.loc[df_dev_captions_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_dev_captions_comp.loc[df_dev_captions_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_dev_captions_comp.loc[df_dev_captions_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_dev_captions_comp.loc[df_dev_captions_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_dev_captions_comp.loc[df_dev_captions_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_dev_captions_comp.loc[
        df_dev_captions_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_dev_captions_comp.loc[
    ~df_dev_captions_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("dev - captions")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/dev_captions.png", bbox_inches="tight")
plt.show()

# para genero forum

df_res_train_forum = res_train_forum.reset_index()
df_res_train_forum_consw = df_res_train_forum[~df_res_train_forum['index'].str.contains('nosw_')]
df_res_train_forum_consw.loc[:, 'clave'] = df_res_train_forum_consw['index']
df_res_train_forum_consw.loc[:, 'corr_consw'] = df_res_train_forum_consw['num2']
df_res_train_forum_nosw = df_res_train_forum[df_res_train_forum['index'].str.contains('nosw_')]
df_res_train_forum_nosw.loc[:, 'clave'] = df_res_train_forum_nosw['index'].str[5:]
df_res_train_forum_nosw.loc[:, 'corr_nosw'] = df_res_train_forum_nosw['num2']
df_train_forum_comp = pd.merge(df_res_train_forum_consw[['clave', 'corr_consw']],
                               df_res_train_forum_nosw[['clave', 'corr_nosw']],
                               on=['clave'], how='inner')

df_train_forum_comp.loc[df_train_forum_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_train_forum_comp.loc[df_train_forum_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_train_forum_comp.loc[df_train_forum_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_train_forum_comp.loc[df_train_forum_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_train_forum_comp.loc[df_train_forum_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_train_forum_comp.loc[
        df_train_forum_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_train_forum_comp.loc[
    ~df_train_forum_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("train - forum")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/train_forum.png", bbox_inches="tight")
plt.show()

df_res_dev_forum = res_dev_forum.reset_index()
df_res_dev_forum_consw = df_res_dev_forum[~df_res_dev_forum['index'].str.contains('nosw_')]
df_res_dev_forum_consw.loc[:, 'clave'] = df_res_dev_forum_consw['index']
df_res_dev_forum_consw.loc[:, 'corr_consw'] = df_res_dev_forum_consw['num2']
df_res_dev_forum_nosw = df_res_dev_forum[df_res_dev_forum['index'].str.contains('nosw_')]
df_res_dev_forum_nosw.loc[:, 'clave'] = df_res_dev_forum_nosw['index'].str[5:]
df_res_dev_forum_nosw.loc[:, 'corr_nosw'] = df_res_dev_forum_nosw['num2']
df_dev_forum_comp = pd.merge(df_res_dev_forum_consw[['clave', 'corr_consw']],
                             df_res_dev_forum_nosw[['clave', 'corr_nosw']],
                             on=['clave'], how='inner')

df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains('vw'), 'color'] = 'blue'
df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains('semcor'), 'color'] = 'red'
df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains('brown'), 'color'] = 'orange'
df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains('wordnet'), 'color'] = 'black'
df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains('score_aligned'), 'color'] = 'green'

fig, ax = plt.subplots()
formas = {'rel': 'x', 'mih|milha': '^'}
for tipo in ['rel', 'mih|milha']:
    df_plot = df_dev_forum_comp.loc[df_dev_forum_comp['clave'].str.contains(tipo), ['corr_consw', 'corr_nosw', 'color']]
    plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, marker=formas[tipo], s=50)
df_plot = df_dev_forum_comp.loc[
    ~df_dev_forum_comp['clave'].str.contains('rel|mih|milha'), ['corr_consw', 'corr_nosw', 'color']]
plt.scatter(df_plot['corr_consw'], df_plot['corr_nosw'], c=df_plot.color.values, s=50)
plt.plot([-1, 1], [-1, 1], linestyle='dashed')
plt.title("dev - forum")
plt.xlim([-0.25, 0.8])
plt.ylim([-0.25, 0.8])
plt.xticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.yticks(ticks=[-0.25, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
plt.xlabel("Corr con stop words")
plt.ylabel("Corr sin stop words")
plt.grid()
plt.xticks(rotation=30)
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/dev_forum.png", bbox_inches="tight")
plt.show()

# umbrales
markers = {'rel_dist': 'o', 'rel_dot': '>', 'rel_cosin': '<', 'rel_res': 'v', 'rel_lch': '^', 'rel_wup': 's',
           'rel_lin': 'x', 'rel_jcn': 'd', 'rel_path': 'P'}
lines = {'nosw': 'solid', 'consw': 'dotted'}
colors = {'brown': 'green', 'semcor': 'blue', 'wordnet': 'red'}

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='black', linestyle='solid', label='Con stop words'),
                   Line2D([0], [0], color='black', linestyle='dotted', label='Sin stop words'),
                   Line2D([0], [0], color='green', lw=4, label='brown IC'),
                   Line2D([0], [0], color='blue', lw=4, label='semcor IC'),
                   Line2D([0], [0], color='red', lw=4, label='wordnet IC'),
                   Line2D([0], [0], color='black', lw=4, label='W2V'),
                   Line2D([0], [0], marker='o', markersize=4, label='W2V - distancia',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='>', markersize=4, label='W2V - dot prod',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='<', markersize=4, label='W2V - cosin',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='v', markersize=4, label='Resnik',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='^', markersize=4, label='Leancock-Chodorow',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='s', markersize=4, label='Wu-Palmer',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='x', markersize=4, label='Lin',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='d', markersize=4, label='Jian-Corath',
                          markerfacecolor='b'),
                   Line2D([0], [0], marker='P', markersize=4, label='Shortest Path')
                   ]

plt.rcParams["figure.figsize"] = (10.5, 6.4)
# para general

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_train_comp[df_train_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_train_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_train_comp = pd.concat(
            [df_umbrales_train_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_train_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    if color_plot == "red":
        plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'], linewidth=3)
        plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'], linewidth=3)
    else:
        plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
        plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("train - general")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_train_general.png", bbox_inches='tight')
plt.show()
plt.clf()
list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_dev_comp[df_dev_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_dev_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_dev_comp = pd.concat(
            [df_umbrales_dev_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_dev_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    if color_plot == "red":
        plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'], linewidth=3)
        plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'], linewidth=3)
    else:
        plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
        plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])
plt.xlim([0.5, 0.9])
plt.title("dev - general")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_dev_general.png", bbox_inches='tight')
plt.show()
plt.clf()

# para genero news

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_train_news_comp[df_train_news_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_train_news_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_train_news_comp = pd.concat(
            [df_umbrales_train_news_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_train_news_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("train - news")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_train_news.png", bbox_inches='tight')
plt.show()
plt.clf()

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_dev_news_comp[df_dev_news_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_dev_news_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_dev_news_comp = pd.concat(
            [df_umbrales_dev_news_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_dev_news_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("dev - news")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_dev_news.png", bbox_inches='tight')
plt.show()
plt.clf()

# para genero captions

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_train_captions_comp[df_train_captions_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_train_captions_comp = df_tmp[
            ['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_train_captions_comp = pd.concat(
            [df_umbrales_train_captions_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_train_captions_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("train - captions")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_train_captions.png", bbox_inches='tight')
plt.show()
plt.clf()

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_dev_captions_comp[df_dev_captions_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_dev_captions_comp = df_tmp[
            ['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_dev_captions_comp = pd.concat(
            [df_umbrales_dev_captions_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_dev_captions_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("dev - captions")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_dev_captions.png", bbox_inches='tight')
plt.show()
plt.clf()

# para genero forum
list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_train_forum_comp[df_train_forum_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_train_forum_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_train_forum_comp = pd.concat(
            [df_umbrales_train_forum_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_train_forum_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("train - forum")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_train_forum.png", bbox_inches='tight')
plt.show()
plt.clf()

list_umbrales = [0.5, 0.75, 0.85, 0.9]
for i, um in enumerate(list_umbrales):
    df_tmp = df_dev_forum_comp[df_dev_forum_comp["clave"].str.contains(str(um))]
    df_tmp.loc[:, 'clave_tmp'] = df_tmp.clave.apply(lambda x: "_".join(x.split("_")[:-1]))
    df_tmp.loc[:, 'corr_consw_' + str(um)] = df_tmp.loc[:, 'corr_consw']
    df_tmp.loc[:, 'corr_nosw_' + str(um)] = df_tmp.loc[:, 'corr_nosw']
    if i == 0:
        df_umbrales_dev_forum_comp = df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index(
            'clave_tmp')
    else:
        df_umbrales_dev_forum_comp = pd.concat(
            [df_umbrales_dev_forum_comp,
             df_tmp[['clave_tmp', 'corr_consw_' + str(um), 'corr_nosw_' + str(um)]].set_index('clave_tmp')], axis=1)

for row in df_umbrales_dev_forum_comp.iterrows():
    list_val_nosw = []
    list_val_consw = []
    color_plot = "black"
    for um in list_umbrales:
        list_val_nosw.append(row[1][[x for x in row[1].index if 'nosw' in x and str(um) in x][0]])
        list_val_consw.append(row[1][[x for x in row[1].index if 'consw' in x and str(um) in x][0]])
    for mark in markers.items():
        if mark[0] in row[1].name:
            marcador = mark[1]
    for color in colors.items():
        if color[0] in row[1].name:
            color_plot = color[1]

    print(row[1].name, color[0], color[0] in row[1].name, color_plot)
    plt.plot(list_umbrales, list_val_nosw, color=color_plot, marker=marcador, linestyle=lines['nosw'])
    plt.plot(list_umbrales, list_val_consw, color=color_plot, marker=marcador, linestyle=lines['consw'])

plt.xlim([0.5, 0.9])
plt.title("dev - forum")
plt.ylabel(" Correlación ")
plt.xlabel("Umbral")
plt.legend(handles=legend_elements, bbox_to_anchor=(0, 1.04, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=5)
plt.savefig("res/um_dev_forum.png", bbox_inches='tight')
plt.show()
plt.clf()
