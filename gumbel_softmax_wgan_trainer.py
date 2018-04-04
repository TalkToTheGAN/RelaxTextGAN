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
from models.discriminator import LSTMWGANDiscriminator as Discriminator
from utils import Utils
from data_loader import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
print(opt)


total_epochs = 40
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

        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size, 1.0)
        fake_target = Variable(torch.zeros((data.size(0), 1)))

        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)


        # D_real_loss = criterion(real_pred, real_target)
        # D_fake_loss = criterion(fake_pred, fake_target)
        # D_loss = D_real_loss + D_fake_loss
        D_loss = -(torch.mean(torch.log(real_pred) + torch.log(1- fake_pred)))
        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)


        fake_data = generator.relaxed_sample(batch_size, g_seq_length, vocab_size, 1.0)
        fake_pred = discriminator(fake_data)

        G_loss = -torch.mean(torch.log(fake_data))
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        total_G_loss += G_loss.data[0]
        total_D_loss += D_loss.data[0]

        all_G_losses.append(total_G_loss)
        all_D_losses.append(total_D_loss)

        if (count%50 == 0):
            sample = generator.sample(1, g_seq_length)
            # print('fake_data.size():', fake_data.size())
            fake_data = convert_to_one_hot(sample.view(1,g_seq_length), vocab_size)
            # print('fake_data.size():', fake_data.size())
            real_data = data[0].resize(1, g_seq_length)
            real_data = convert_to_one_hot(real_data, vocab_size)
            fake_str = data_loader.convert_to_char(sample.view(1, g_seq_length))
            real_str = data_loader.convert_to_char(data[0].view(1, g_seq_length))
            fake_pred = discriminator(fake_data)
            real_pred = discriminator(real_data)

            D_loss = -(torch.mean(torch.log(real_pred) + torch.log(1- fake_pred)))
            G_loss = -torch.mean(torch.log(real_pred))

            # print(torch.log(real_pred))
            print('real_data and real_pred:', real_str, real_pred.data[0, 0])
            print('fake_data and fake_pred:', fake_str, fake_pred.data[0, 0])
            print('D_total_loss', D_loss.data[0])
            print('G loss', G_loss.data[0])
            print('--------------')

        count+=1
    
    print("Total G loss: ", total_G_loss/count)
    print("Total D loss: ", total_D_loss/count)

    data_loader.reset()
    return all_G_losses, all_D_losses

def main():

    data_loader = DataLoader(data_path, batch_size)
    generator = Generator(vocab_size, g_emb_dim, g_hidden_dim)
    discriminator = Discriminator(vocab_size, d_hidden_dim)
    
    gen_optimizer = optim.RMSprop(generator.parameters(), lr=0.00001)
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr = 0.00001)

    bce_criterion = nn.BCELoss()

    if (opt.cuda):
        generator.cuda()


    all_G_losses = []
    all_D_losses = []
    for i in tqdm(range(total_epochs)):
        g_losses, d_losses = train_gan_epoch(discriminator, generator, data_loader, gen_optimizer, disc_optimizer, bce_criterion)
        all_G_losses += g_losses
        all_D_losses += d_losses

        sample = generator.sample(batch_size, g_seq_length)
        print("SAMPLE SIZE:", sample.size())
        all_strings = []
        for each_str in data_loader.convert_to_char(sample):
            all_strings.append(each_str)
        print(all_strings)
        print("Goodness string:", Utils.get_data_goodness_score([all_strings]))


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