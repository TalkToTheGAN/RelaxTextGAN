import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from models.generator import Generator
from models.discriminator import LSTMDiscriminator as Discriminator
from utils import Utils
from data_loader import DataLoader

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


def train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer, criterion):
    count = 0
    all_G_losses = []
    all_D_losses = []
    all_G_rewards = []
    for data, _ in tqdm(data_loader):
        total_G_loss = 0.0
        total_D_loss = 0.0
        target = torch.ones(data.size())
        data = Variable(data)       #dim=batch_size x sequence_length e.g: 16x15
        target = Variable(target)   #dim=batch_size x sequence_length e.g: 16x15
        if opt.cuda:
            data, target = data.cuda(), target.cuda()

        real_data = convert_to_one_hot(data, vocab_size)
        real_target = Variable(torch.ones((data.size(0), 1)))

        fake_data = generator.sample_one_hot(batch_size, g_seq_length, vocab_size)
        fake_target = Variable(torch.zeros((data.size(0), 1)))

        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)


        # --------------D Trainer ---------------------
        D_real_loss = criterion(real_pred, real_target)
        D_fake_loss = criterion(fake_pred, fake_target)
        D_loss = D_real_loss + D_fake_loss
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        #----------------G Trainer --------------------

        gen_optimizer.zero_grad()
        fake_data, actual_log_probs, sampled_log_probs = generator.sample_one_hot_with_prob(batch_size, g_seq_length, vocab_size)
        fake_pred = discriminator(fake_data)
        total_reward = fake_pred.sum(dim=0)
        sampled_log_probs*=total_reward

        for batch_index, each in enumerate(sampled_log_probs):
            for seq_index, each_seq in enumerate(each):
                each_seq.backward(sampled_log_probs[batch_index, seq_index])

        gen_optimizer.step()

        all_G_rewards.append(total_reward)
        count+=1

    data_loader.reset()
    return 


def main():

    data_loader = DataLoader(data_path, batch_size)
    generator = Generator(vocab_size, g_emb_dim, g_hidden_dim)
    discriminator = Discriminator(vocab_size, d_hidden_dim)
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0001)

    bce_criterion = nn.BCELoss()

    if (opt.cuda):
        generator.cuda()
        discriminator.cuda()

    for i in tqdm(range(total_epochs)):
        print("EPOCH:", i)
        all_G_rewards = train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer, bce_criterion)

        if(i%3==0):
            with open('./data/reinforce_gan_data_epoch'+ str(i) + '.txt', 'w') as f:
                for each_str in data_loader.convert_to_char(sample):
                    f.write(each_str+'\n')

    sample = generator.sample(batch_size, g_seq_length)

    with open('./data/reinforce_gan_final_data.txt', 'w') as f:
        for each_str in data_loader.convert_to_char(sample):
            f.write(each_str+'\n')

    plt.plot(all_G_rewards)
    plt.show()
if __name__ == '__main__':
    main()