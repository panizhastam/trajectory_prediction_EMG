import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import MyData
import numpy as np
from tqdm import tqdm

# ['emg_elbow','emg_shfe','emg_shaa','pos_elbow','pos_shfe','pos_shaa']


# df1 = pd.read_csv('dataset/emgdata1.csv')
# # df1 = df1.drop(['t'], axis=1)
# df2 = pd.read_csv('dataset/emgdata2.csv')
# df2 = df2.drop(columns=['t'])
# df3 = pd.read_csv('dataset/emgdata3.csv')
# df3 = df3.drop(['t'], axis=1)
# df4 = pd.read_csv('dataset/emgdata4.csv')
# df4 = df4.drop(['t'], axis=1)

# df5 = pd.concat([df1, df2, df3, df4], ignore_index=True)
# df5.to_csv('dataset/allemgdata.csv', index=False)

df = pd.read_csv('dataset/allemgdata.csv')
df['pos_elbow'] = df['pos_elbow'] - df['pos_elbow'].min()
df['pos_shfe'] = df['pos_shfe'] - df['pos_shfe'].min()
df['pos_shaa'] = df['pos_shaa'] - df['pos_shaa'].min()


# features = df[['emg_elbow','emg_shfe','emg_shaa','pos_elbow','pos_shfe','pos_shaa']].to_numpy().tolist()
# target = df[['pos_elbow','pos_shfe','pos_shaa']].to_numpy().tolist()

train, test = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)
test.to_csv('dataset/testdata.csv', index=False)


train, validation = train_test_split(train, test_size=0.1, shuffle=False,random_state=42)
train.to_csv('dataset/traindata.csv', index=False)
validation.to_csv('dataset/validationdata.csv', index=False)
print("The train size is: ", len(train))

batch_size = 25
window_size = 50

path = 'dataset/traindata.csv'
train_data = MyData.PrepareDataset(path,window_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print("The tensor shape in a training set is: ", len(train_loader) * batch_size)

path = 'dataset/testdata.csv'
test_data = MyData.PrepareDataset(path,window_size)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
print("The tensor shape in a test set is: ", len(test_loader) * batch_size)

path = 'dataset/validationdata.csv'
valid_data = MyData.PrepareDataset(path,window_size)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
print("The tensor shape in a valid set is: ", len(valid_loader) * batch_size)


print("done!")








