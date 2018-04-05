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
batch_size = 64
data_path = './data/math_equation_data.txt'
g_seq_length = 15
g_emb_dim = 8
g_hidden_dim = 8
d_hidden_dim = 8
g_output_dim = 1
vocab_size = 6
checkpoint_dir = './checkpoints'

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

def goodness_score_metric(generator, data_loader, print_sample = False):
    sample = generator.sample(batch_size, g_seq_length)
    all_strings = []
    for each_str in data_loader.convert_to_char(sample):
        all_strings.append(each_str)

    if(print_sample==True):
        for each_str in all_strings:
            print(each_str)

    print("Goodness string:", Utils.get_data_goodness_score([all_strings]))

def pretrain_lstm(generator, data_loader, optimizer, criterion, epochs):
    for i in range(epochs):
        for data, target in data_loader:
            data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
            target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            pred = generator(data)

            target = target.view(-1)
            pred = pred.view(-1, vocab_size)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        data_loader.reset()
    
    goodness_score_metric(generator, data_loader, True)
    
    print('Finished pretraining LSTM')


def train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer, criterion):
    count = 0
    all_G_losses = []
    all_D_losses = []
    for data, _ in data_loader:
        total_G_loss = 0.0
        total_D_loss = 0.0
        target = torch.ones(data.size())
        data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
        target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
        if opt.cuda:
            data, target = data.cuda(), target.cuda()

        real_data = convert_to_one_hot(data, vocab_size)
        real_target = Variable(torch.ones((data.size(0), 1)))

        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size, 2)
        fake_target = Variable(torch.zeros((data.size(0), 1)))

        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)


        D_real_loss = criterion(real_pred, real_target)
        D_fake_loss = criterion(fake_pred, fake_target)
        D_loss = D_real_loss + D_fake_loss
        # D_loss = -0.5 * torch.log(real_pred.sum(dim=0)) - 0.5 * torch.log(1 - fake_pred.sum(dim=0))
        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size, 2)
        fake_pred = discriminator(fake_data)

        G_loss = criterion(fake_pred, real_target)
        # G_loss = -0.5 * torch.log(fake_pred.sum(dim=0))
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        total_G_loss += G_loss.data[0]
        total_D_loss += D_loss.data[0]

        all_G_losses.append(total_G_loss)
        all_D_losses.append(total_D_loss)

        count+=1
    
    print("Total G loss: ", total_G_loss/count)
    print("Total D loss: ", total_D_loss/count)

    data_loader.reset()
    return all_G_losses, all_D_losses

def main():

    data_loader = DataLoader(data_path, batch_size)
    generator = Generator(vocab_size, g_emb_dim, g_hidden_dim)
    discriminator = Discriminator(vocab_size, d_hidden_dim)
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0001)

    bce_criterion = nn.BCELoss()
    gen_criterion = nn.NLLLoss(size_average=False)

    if (opt.cuda):
        generator.cuda()

    pretrain_lstm(generator, data_loader, gen_optimizer, gen_criterion, 10)

    all_G_losses = []
    all_D_losses = []
    for i in tqdm(range(total_epochs)):
        g_losses, d_losses = train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer, bce_criterion)
        all_G_losses += g_losses
        all_D_losses += d_losses


    sample = generator.sample(batch_size, g_seq_length)

    print(generator)

    with open('./data/gumbel_softmax_gan_gen.txt', 'w') as f:
        for each_str in data_loader.convert_to_char(sample):
            f.write(each_str+'\n')

    gen_file_name = 'gen_gumbel_softmax_' + str(total_epochs) + '.pth'
    disc_file_name = 'disc_gumbel_softmax_' + str(total_epochs) + '.pth'

    Utils.save_checkpoints(checkpoint_dir, gen_file_name, generator)
    Utils.save_checkpoints(checkpoint_dir, disc_file_name, discriminator)

    plt.plot(list(range(len(all_G_losses))), all_G_losses, 'g-', label='gen loss')
    plt.plot(list(range(len(all_D_losses))), all_D_losses, 'b-', label='disc loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()