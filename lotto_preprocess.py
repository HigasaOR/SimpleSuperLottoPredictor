import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class SuperLottoDataset(Dataset):
    ''' Super lottery dataset.
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        if self.Y is None: return self.X[idx]
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def trim_to_csv():
    ''' Trim from txt to csv.
    '''
    rdata_list = []
    with open("./data/data.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            rdata = line.split()
            if len(rdata) > 0:
                rdata = rdata[2:9]
                rdata_list.append(rdata)
            line = f.readline()
    with open('./data/numbers.csv','w', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(rdata_list)


def get_one_hot_dataframe_from_csv():
    ''' Gets one hot encoding pd dataframe from preprocessed csv.
    '''
    df = pd.read_csv('./data/numbers.csv', header=None)
    oht = pd.get_dummies(df[0])
    for i in range(1, 7):
        oht = pd.concat([oht, pd.get_dummies(df[i])], axis=1, ignore_index=True)
    return oht


def make_dataset(oht):
    ''' Build the super lotto dataset from one hot pd dataframe.
    '''
    oht_np = np.array(oht)
    # print(oht_np.shape)

    X_np = oht_np[:oht_np.shape[0]-1]
    Y_np = oht_np[1:]

    X_t = torch.FloatTensor(X_np)
    Y_t = torch.LongTensor(Y_np)
    print(X_t.shape)

    dataset = SuperLottoDataset(X=X_t, Y=Y_t)
    return dataset
