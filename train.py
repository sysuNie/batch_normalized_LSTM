# -*- coding: UTF-8 -*-
import argparse
import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from torchvision import datasets, transforms
from torch.autograd import Variable
from batch_normalization_LSTM import LSTMCell, BNLSTMCell, LSTM

parser = argparse.ArgumentParser(description='PyTorch train BNLSTM model on MNIST dataset')

parser.add_argument('--data', type=str, default='data/',
                    help='The path to save MNIST dataset')
parser.add_argument('--model', required=True, choices=['lstm', 'bnlstm'],
                    help='The name of a model to use')
parser.add_argument('--save', type=str, default='log/',
                    help='The path to save model files')
parser.add_argument('--hidden-size', type=int, default=1000,
                    help='The number of hidden unit size')
parser.add_argument('--batch-size', type=int, default=128,
                    help='The size of each batch')
parser.add_argument('--epoches', type=int, default=20,
                    help='The iteration count')
parser.add_argument('--cuda', default=False, action='store_true',
                    help='The value specifying whether to use GPU')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.cuda)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = datasets.MNIST(
        root=args.data, train=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                      ]), download=True)
test_dataset = datasets.MNIST(
        root=args.data, train=False,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                      ]), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True, pin_memory=True)

test_dataset = DataLoader(dataset=test_dataset,
                          batch_size=args.batch_size,
                          shuffle=True, pin_memory=True)

model_name = args.model
if model_name == 'bnlstm':
    model = LSTM(cell_class=BNLSTMCell, input_size=28,
                 hidden_size=args.hidden_size, batch_first=True, max_length=784)
elif model_name == 'lstm':
    model = LSTM(cell_class=LSTMCell, input_size=28,
                 hidden_size=args.hidden_size, batch_first=True)
else:
    raise ValueError

fc = nn.Linear(in_features=args.hidden_size, out_features=10)
criterion = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(fc.parameters())
optimizer = optim.SGD(params=params, lr=1e-3, momentum=0.9)

def computer_loss(data, label):
    h0 = Variable(data.data.new(data.size(0), args.hidden_size).normal_(0, 0.1))
    c0 = Variable(data.data.new(data.size(0), args.hidden_size).normal_(0, 0.1))
    hx = (h0, c0)

    _, (h_n, _) = model(input_=data, hx=hx)
    logits = fc(h_n[0])
    loss = criterion(input=logits, target=label)
    return loss

for epoch in range(args.epoches):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if args.cuda():
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        optimizer.zero_grad()
        loss = computer_loss(data=images, labels=labels)
        loss.backend()
        optimizer.step()

        if(i+1) % 100 == 0:
            if (i + 1) % 100 == 0:
                print ('Epoch [%d/%d],Iter [%d/%d] Loss: %.4f'
                       % (epoch + 1, args.epoches, i + 1, len(train_dataset) // args.batch_size,
                          loss.data[0]))
