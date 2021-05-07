from ubyte import UByte
from fastai import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from tqdm import tqdm, trange

def create_train_dl(read=10,batch_size=256):
    images = []
    labels = []
    with UByte('train-images-idx3-ubyte.gz', read=read) as d:
        images = d.data
        # print('Images: ', d.count)

    with UByte('train-labels-idx1-ubyte.gz', mode='l', read=read) as d:
        labels = d.data
        # print('Labels: ', d.count)

    images = torch.tensor(images).reshape(-1, 28*28)
    dset = list(zip(images, torch.tensor(format_labels(labels))))
    return DataLoader(dset, batch_size=batch_size)


def create_test_dl(read=10, batch_size=256):
    images = []
    labels = []
    with UByte('t10k-images-idx3-ubyte.gz', read=read) as d:
        images = d.data
        # print('Images: ', d.count)

    with UByte('t10k-labels-idx1-ubyte.gz', mode='l', read=read) as d:
        labels = d.data
        # print('Labels: ', d.count)

    images = torch.tensor(images).reshape(-1, 28*28)
    dset = list(zip(images, torch.tensor(format_labels(labels))))
    return DataLoader(dset, batch_size=batch_size)

# labels are 1,2,3 floats, we need them to be 1-hot arrays
def format_labels(l):
    labels = np.empty((len(l), 10)) 
    for i in range(len(l)):
        labels[i] = np.eye(1, 10, int(l[i]))

    return labels

def create_params(s):
    return torch.randn(s).requires_grad_()

def create_params_l1():
    w1 = create_params((28*28, 60))
    b1 = create_params(1)
    w2 = create_params((60, 10))
    b2 = create_params(1)
    return (w1, b1, w2, b2)

def create_model(params):
    w1, b1, w2, b2 = params
    def model1(x):
        l1 = torch.sigmoid(x@w1 + b1)
        return l1@w2 + b2

    return model1

mse_loss = nn.MSELoss()

def calc_loss(preds, target):
    preds = preds.sigmoid()
    return mse_loss(preds.float(), target.float())

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = calc_loss(preds, yb)
    loss.backward()

def train_epoch(dl, model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
    
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    return 1 - (yb - preds).mean()
    
def validate_epoch(model, dset):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in dset]
    return round(torch.stack(accs).mean().item(), 4)

def run():
    train_dl = create_train_dl(10000)
    test_dl = create_test_dl(1000)
    params = create_params_l1()
    model = create_model(params)

    pbar = trange(10000)
    for i in pbar:
        train_epoch(train_dl, model, 1e-3, params)
        pbar.set_postfix(acc=validate_epoch(model, test_dl))

    # Visual validaition
    for test_images, test_labels in test_dl:  
        for i in range(2):
            sample_image = test_images[i]
            sample_label = test_labels[i]
            result = model(sample_image)
            print(torch.argmax(result), torch.argmax(sample_label))


run()    

