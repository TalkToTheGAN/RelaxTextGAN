import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from models.generator import Generator
from models.discriminator import LSTMDiscriminator as Discriminator
from utils import Utils
from data_loader import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)


total_epochs = 20
batch_size = 16
data_path = './data/math_equation_data.txt'
g_seq_length = 15
g_emb_dim = 8
g_hidden_dim = 8
d_hidden_dim = 8
g_output_dim = 1
vocab_size = 7  # need to not hard code this. Todo for later.

def convert_to_one_hot(data, vocab_size):
    """
        data dims: (batch_size, seq_len)
        returns:(batch_size, seq_len, vocab_size)
    """
    batch_size = data.size(0)
    seq_len = data.size(1)

    samples = Variable(torch.Tensor(batch_size, seq_len, vocab_size))
    one_hot = Variable(torch.zeros((batch_size, vocab_size)).long())

    for i in range(batch_size):
        x = data[i].view(-1,1)
        one_hot = Variable(torch.zeros((seq_len, vocab_size)).long())
        samples[i] = one_hot.scatter_(1, x, 1)

    return samples

def train_generator_epoch(model, data_loader, criterion, optimizer):
    losses_array = []
    for data, target in data_loader:
        total_loss = 0.0
        total_words = 0.0
        data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
        target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        pred = model(data)

        target = target.view(-1)
        pred = pred.view(-1, vocab_size)

        loss = criterion(pred, target)
        total_loss += loss.data[0]
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_array.append(total_loss)

    data_loader.reset()


def train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer):
    for data, _ in data_loader:
        total_loss = 0.0
        total_words = 0.0
        target = torch.ones(data.size())
        data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
        target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
        if opt.cuda:
            data, target = data.cuda(), target.cuda()

        real_data = convert_to_one_hot(data, vocab_size)
        real_target = torch.ones((data.size(0), 1))

        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size)
        fake_target = torch.zeros((data.size(0), 1))

        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)


        D_loss = -0.5 * torch.log(real_pred.sum(dim=0)) - 0.5 * torch.log(1 - fake_pred.sum(dim=0))
        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size)
        fake_pred = discriminator(fake_data)

        G_loss = -0.5 * torch.log(fake_pred.sum(dim=0))
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

    data_loader.reset()

def main():

    data_loader = DataLoader(data_path, batch_size)
    generator = Generator(vocab_size, g_emb_dim, g_hidden_dim)
    discriminator = Discriminator(vocab_size, d_hidden_dim)
    
    gen_optimizer = optim.Adam(generator.parameters())
    gen_criterion = nn.NLLLoss(size_average=False)

    disc_optimizer = optim.Adam(discriminator.parameters())

    if (opt.cuda):
        generator.cuda()


    for i in tqdm(range(20)):
        train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer)


    sample = generator.sample(batch_size, g_seq_length)

    with open('./data/gumbel_softmax_gan_gen.txt', 'w') as f:
        for each_str in data_loader.convert_to_char(sample):
            f.write(each_str+'\n')


if __name__ == '__main__':
    main()