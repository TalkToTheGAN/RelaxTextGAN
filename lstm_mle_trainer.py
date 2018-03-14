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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)


# all constants
total_epochs = 20
batch_size = 16
data_path = './data/math_equation_data.txt'
g_seq_length = 15
g_emb_dim = 8
g_hidden_dim = 8
vocab_size = 7  # need to not hard code this. Todo for later.


def main():

    data_loader = DataLoader(data_path, batch_size)
    generator = PlainLSTM(vocab_size, g_emb_dim, g_hidden_dim)
    optimizer = optim.Adam(generator.parameters())
    if (opt.cuda):
        generator.cuda()

    losses_array = []
    for epoch in tqdm(range(total_epochs)):
        for data, target in data_loader:

            total_loss = 0.0
            total_words = 0.0
            data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
            target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            pred = generator(data)

            target = target.view(-1)
            pred = pred.view(-1, vocab_size)

            gen_criterion = nn.NLLLoss(size_average=False)
            loss = gen_criterion(pred, target)
            total_loss += loss.data[0]
            total_words += data.size(0) * data.size(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_array.append(total_loss)

        data_loader.reset()

    sample = generator.sample(batch_size, g_seq_length)

    with open('./data/lstm_mle_gen_data.txt', 'w') as f:
        for each_str in data_loader.convert_to_char(sample):
            f.write(each_str+'\n')
    plt.plot(losses_array)
    plt.show()


if __name__ == '__main__':
    main()