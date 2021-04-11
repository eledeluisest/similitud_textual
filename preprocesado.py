import pandas as pd

df_train = pd.DataFrame()
for i in range(0,8):
    df_read = pd.read_csv("data/train_preprocessed"+str(i)+".csv", sep=';')
    if i == 0:
        df_train = df_read
    else:
        df_train = pd.concat([df_train, df_read], axis=0)

df_train['len_or1'] = df_train['s1_tokenized'].apply(lambda x: len(x.split(',')))
df_train['len_or2'] = df_train['s2_tokenized'].apply(lambda x: len(x.split(',')))
df_train['len_or1_nosw'] = df_train['s1_tok_nosw'].apply(lambda x: len(x.split(',')))
df_train['len_or2_nosw'] = df_train['s2_tok_nosw'].apply(lambda x: len(x.split(',')))

df_train.to_csv("data/train_features.csv", sep=';', index=False)

df_dev = pd.DataFrame()
for i in range(0,4):
    df_read = pd.read_csv("data/dev_preprocessed"+str(i)+".csv", sep=';')
    if i == 0:
        df_dev = df_read
    else:
        df_dev = pd.concat([df_dev, df_read], axis=0)
df_dev['len_or1'] = df_dev['s1_tokenized'].apply(lambda x: len(x.split(',')))
df_dev['len_or2'] = df_dev['s2_tokenized'].apply(lambda x: len(x.split(',')))
df_dev['len_or1_nosw'] = df_dev['s1_tok_nosw'].apply(lambda x: len(x.split(',')))
df_dev['len_or2_nosw'] = df_dev['s2_tok_nosw'].apply(lambda x: len(x.split(',')))

df_dev.to_csv("data/dev_features.csv", sep=';', index=False)

df_test = pd.DataFrame()
for i in range(0,4):
    df_read = pd.read_csv("data/test_preprocessed"+str(i)+".csv", sep=';')
    if i == 0:
        df_test = df_read
    else:
        df_test = pd.concat([df_test, df_read], axis=0)

df_test['len_or1'] = df_test['s1_tokenized'].apply(lambda x: len(x.split(',')))
df_test['len_or2'] = df_test['s2_tokenized'].apply(lambda x: len(x.split(',')))
df_test['len_or1_nosw'] = df_test['s1_tok_nosw'].apply(lambda x: len(x.split(',')))
df_test['len_or2_nosw'] = df_test['s2_tok_nosw'].apply(lambda x: len(x.split(',')))

df_test.to_csv("data/test_features.csv", sep=';', index=False)

