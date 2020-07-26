import os
from operator import itemgetter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

from lotto_preprocess import get_one_hot_dataframe_from_csv, make_dataset


class SuperLottoPredictor(nn.Module):
    # Simple MLP model

    def __init__(self, num_features):
        super(SuperLottoPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            #######################
            nn.Dropout(),
            #######################
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            #######################
            nn.Dropout(),
            #######################
            nn.Linear(1024, num_features)
        )

    def forward(self, x):
        return self.fc(x)


def cross_entropy_loss(input, target, size_average=True):
    # Using log softmax + NLL loss because pytorch's crossentropy
    # doesn't support one hot encoding.
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def get_lotto_numbers_from_tensor(pred, n):
    '''  Gets first n lotto numbers from prediction tensor vector.
         Returns a tuple list, with tup[0] the number region and tup[1] the number.
    '''
    pred = pred.cpu()
    _, indices = torch.sort(pred, descending=True)
    indices = indices.flatten()[:n]
    num_list = []
    for i in range(n):
        idx = indices[i].numpy() + 1
        ch = 2 if idx // 38 >= 6 else 1
        num = idx % 38
        num_list.append((ch, num))

    num_list.sort(key=itemgetter(0, 1))
    return num_list


if __name__ == "__main__":

    # some settings
    num_features = 236  # 38 * 6 + 8
    batch_size = 16
    num_epoch = 150

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make dataset
    oht = get_one_hot_dataframe_from_csv()
    last_row = torch.FloatTensor(np.array(oht.tail(1))).to(device)

    sp_dataset = make_dataset(oht)

    # make dataloader
    print("Making dataloader...")
    sp_loader = DataLoader(dataset=sp_dataset, batch_size = batch_size,
                            shuffle = False, num_workers = 8)

    # prepare model
    print("Preparing model...")
    model = SuperLottoPredictor(num_features)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    # start training
    model.train()

    for epoch in range(num_epoch):
        epoch_start_time = time.time()

        train_tot_loss = 0.0

        for i, d in enumerate(sp_loader):  # iterate data loader
            batch_x = d[0].to(device)
            batch_y = d[1].to(device)

            # foward pass
            train_pred = model(batch_x)  # prediction
            batch_loss = cross_entropy_loss(train_pred, batch_y)
            # batch_loss = loss(train_pred, batch_y)  # calculate loss

            # backward pass
            optimizer.zero_grad()  # zero all the gradients
            batch_loss.backward()  # back propagation, calculate gradients
            optimizer.step()  # update param values using optimizer

            train_tot_loss += batch_loss.item()

        write_string = "[%03d/%03d] %2.2f sec(s). Average Train Loss: %3.6f ." % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
            train_tot_loss/sp_dataset.__len__())

        print(write_string)

    # save model
    saving_path = "./model/my_predictor.mdl"
    torch.save(model.state_dict(), saving_path)

    # prediction
    model.eval()  # turn model to evaluation mode

    pred = model(last_row)
    # tuple list, with tup[0] the number region and tup[1] the number.
    pred_number_list = get_lotto_numbers_from_tensor(pred, 7)

    print("############################################################")
    print(pred_number_list)

    # save prediction to file
    with open('pred.txt', 'w') as f:
        for item in pred_number_list:
            f.write("%s\n" % str(item))
