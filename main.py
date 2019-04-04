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

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

loss_func = dice_loss

train_dl = get_data(load_data('Arrays/'), 10)
model, opt = get_model()

fit(5, model, loss_func, opt, train_dl)

