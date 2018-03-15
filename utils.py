import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Utils:
    @staticmethod
    def g_output_prob(prob, BATCH_SIZE, g_sequence_len):
        softmax = nn.Softmax(dim=1)
        theta_prime = softmax(prob)
        theta_prime = torch.sum(theta_prime, dim=0).view((-1,))/(BATCH_SIZE*g_sequence_len)
        return theta_prime

    # performs a Gumbel-Softmax reparameterization of the input
    @staticmethod
    def gumbel_softmax(theta_prime, VOCAB_SIZE, in_log_space = False, cuda=False):
        u = Variable(torch.log(-torch.log(torch.rand(VOCAB_SIZE))))
        if cuda:
            u = u.cuda()
        if(in_log_space):
            z = theta_prime - u
        else:
            z = torch.log(theta_prime) - u
        return z

    # categorical re-sampling exactly as in Backpropagating through the void - Appendix B
    @staticmethod
    def categorical_re_param(theta_prime, VOCAB_SIZE, b, cuda=False):
        v = Variable(torch.rand(VOCAB_SIZE))
        if cuda:
            v = Variable(torch.rand(VOCAB_SIZE)).cuda()
        z_tilde = -torch.log(-torch.log(v)/theta_prime - torch.log(v[b]))
        z_tilde[b] = -torch.log(-torch.log(v[b]))
        return z_tilde

    @staticmethod
    def save_checkpoints(checkpoint_dir, file_name, model):
        path = os.path.join(checkpoint_dir, file_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(model.state_dict(), path)