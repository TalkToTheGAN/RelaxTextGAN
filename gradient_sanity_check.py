import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from models.plain_lstm import PlainLSTM
from utils import Utils
from data_loader import DataLoader
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)

total_epochs = 10
batch_size = 16
data_path = './data/math_equation_data.txt'
g_seq_length = 15
g_emb_dim = 8
g_hidden_dim = 8
vocab_size = 6  # need to not hard code this. Todo for later.


def main():
    data_loader = DataLoader(data_path, batch_size)
    generator = PlainLSTM(vocab_size, g_emb_dim, g_hidden_dim)
    optimizer = optim.Adam(generator.parameters())
    if (opt.cuda):
        generator.cuda()

    losses_array = []


    for epoch in tqdm(range(total_epochs)):
        for data, target in tqdm(data_loader):

            data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
            target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
            if opt.cuda:
                data, target = data.cuda(), target.cuda()

            data = data.view(g_seq_length, batch_size)
            target = target.view(g_seq_length, batch_size)
            h, c = generator.init_hidden(batch_size)
            generator.zero_grad()

            for i in range(g_seq_length):
                pred, h, c = generator.step(data[i].view(batch_size, 1), h, c) # batch_size x voca_size
                gen_criterion = nn.NLLLoss(size_average=False)
                loss = gen_criterion(pred, target[i].view(-1))
                loss.backward(retain_graph = True)

            optimizer.step()
        data_loader.reset()

    sample = generator.sample(batch_size, g_seq_length)

    with open('./data/lstm_mle_gen_data.txt', 'w') as f:
        for each_str in data_loader.convert_to_char(sample):
            f.write(each_str+'\n')



if __name__ == '__main__':
    main()