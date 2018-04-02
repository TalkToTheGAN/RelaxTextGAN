import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from utils import Utils
import copy



class Generator(nn.Module):
    """Generator """
    def __init__(self, vocab_size, emb_dim, hidden_dim, use_cuda=False):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.init_params()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)  # emb dim: (batch_size, seq_len, emb_dim)
        h0, c0 = self.init_hidden(x.size(0))

        # input to lstm dimensions: (batch, seq_len, input_size)
        output, (h, c) = self.lstm(emb, (h0, c0))    # output dim = (batch_size x seq_len, x hidden_dim)
        
        seq_len = output.size()[1]
        batch_size = output.size()[0]

        pred = self.log_softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        pred = pred.view(batch_size, seq_len, self.vocab_size)

        h0, c0 = self.init_hidden(x.size(0)) # removing history

        return pred

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)))
        return pred, h, c


    def init_hidden(self, batch_size):
        # noise distribution fed to G
        h = Variable(torch.randn((1, batch_size, self.hidden_dim)))
        c = Variable(torch.randn((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c
    
    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0.0, 0.02)

    def sample(self, batch_size, seq_len, x=None):
        res = []
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = Variable(torch.zeros((batch_size, 1)).long())
        if self.use_cuda:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output

    def sample_one_hot(self, batch_size, seq_len, vocab_size):
        x = Variable(torch.zeros((batch_size, 1)).long())
        samples = Variable(torch.Tensor(seq_len, batch_size, vocab_size))

        if(self.use_cuda):
            x = x.cuda()
            samples.cuda()
        h, c = self.init_hidden(batch_size)

        for i in range(seq_len):
            output, h, c = self.step(x, h, c)
            x = output.multinomial(1)
            
            one_hot = Variable(torch.zeros((batch_size, vocab_size)).long())
            samples[i] = one_hot.scatter_(1, x, 1)

        samples = samples.view(batch_size, seq_len, vocab_size)
        return samples
    """
        returns relaxed samples drawn from lstm according to concrete distrubution
        returned tensor dimenstions (batch_size, seq_len, vocab_size)
    """
    def relaxed_sample(self, batch_size, seq_len, vocab_size, temp_coeff=1.0):

        x = Variable(torch.zeros((batch_size, 1)).long())
        samples = Variable(torch.Tensor(seq_len, batch_size, vocab_size))

        if(self.use_cuda):
            x = x.cuda()
            samples.cuda()
        h, c = self.init_hidden(batch_size)

        for i in range(seq_len):
            output, h, c = self.step(x, h, c)   # output is a softmax output of shape (batch_size, vocab_size)
            gs = Utils.gumbel_softmax(output, output.size(1))/temp_coeff
            relaxed_sample = F.softmax(gs, dim=1)
            samples[i] = relaxed_sample

        samples = samples.view(batch_size, seq_len, vocab_size)
        return samples

    def sample_one_hot_with_prob(self, batch_size, seq_len, vocab_size):
        x = Variable(torch.zeros((batch_size, 1)).long())
        samples = Variable(torch.Tensor(seq_len, batch_size, vocab_size))
        actual_probs = Variable(torch.Tensor(seq_len, batch_size, vocab_size))
        sampled_probs = Variable(torch.Tensor(seq_len, batch_size, vocab_size))

        if(self.use_cuda):
            x = x.cuda()
            samples.cuda()
        h, c = self.init_hidden(batch_size)

        for i in range(seq_len):
            output, h, c = self.step(x, h, c)
            actual_probs[i] = output
            x = output.multinomial(1)
            
            one_hot = Variable(torch.zeros((batch_size, vocab_size)).long())
            p = Variable(torch.zeros((batch_size, vocab_size)))
            samples[i] = one_hot.scatter_(1, x, 1)

            for output_index, each in enumerate(output):
                required_index = copy.copy(x[output_index].data[0])
                required_prob = output[output_index][required_index].data[0]
                p[output_index, required_index] = np.log(required_prob)
            sampled_probs[i] = p

        samples = samples.view(batch_size, seq_len, vocab_size)
        actual_probs = actual_probs.view(batch_size, seq_len, vocab_size)
        actual_probs = torch.log(actual_probs)
        sampled_probs = sampled_probs.view(batch_size, seq_len, vocab_size)
        return samples, actual_probs, sampled_probs
