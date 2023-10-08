import os
import pandas as pd


train_names = os.listdir('data/train')
train_ratio = 0.8

train_end = int(len(train_names) * train_ratio)
train = train_names[:train_end]
test = train_names[train_end:]


def build_df(names):
    df = []
    for it in names:
        df.append([it, it.replace('.jpg', '_mask.gif')])
    df = pd.DataFrame(df, columns=['image', 'mask'])
    return df

train_df = build_df(train)
test_df = build_df(test)

train_df.to_csv('data/train_desc.csv', index=False)
test_df.to_csv('data/test_desc.csv', index=False)
