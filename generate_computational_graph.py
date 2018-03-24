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
from torchviz import make_dot, make_dot_from_trace

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
    for data, _ in data_loader:

        # params = generator.state_dict()
        # print(generator.named_parameters())
        for x in generator.named_parameters():
            print(x)
        # print(params)
        generator.zero_grad()
        output = generator.test_sample(batch_size, g_seq_length, vocab_size)
        dot = make_dot(output, params = dict(generator.named_parameters()))
        dot.format = 'svg'
        dot.render()
        break

if __name__ == '__main__':
    main()