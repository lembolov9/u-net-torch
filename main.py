import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot
from torch import nn, optim
import torch.nn.functional as F

from model import uNet


def load_data(path):

    X_input = []
    X_target = []

    for i in sorted(os.listdir(path)):
        if (i.find('FLAIR') != -1):
            print(i)
            arr = np.load(path + i)
            if (i.find('train') != -1):
                X_input.append(arr)
            else:
                X_target.append(arr)

    X_input = np.transpose(np.asarray([np.concatenate(X_input, axis=0)]), (1,0,2,3))
    X_target = np.transpose(np.asarray([np.concatenate(X_target, axis=0)]), (1,0,2,3))


    X_input, X_target = map(torch.tensor, (X_input, X_target))
    train_ds = TensorDataset(X_input, X_target)
    return train_ds

def get_data(train_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True)
    )

def get_model():
    return uNet(), optim.Adam(uNet().parameters(), lr=0.025)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    print(loss)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)



def fit(epochs, model, loss_func, opt, train_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1).float()
        m2 = targets.view(num, -1).float()
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

loss_func = SoftDiceLoss()

train_dl = get_data(load_data('Arrays/'), 10)
model, opt = get_model()

fit(5, model, loss_func, opt, train_dl)

