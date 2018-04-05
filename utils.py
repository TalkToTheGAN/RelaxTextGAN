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


    @staticmethod
    def get_data_goodness_score(all_data):
        '''
        Per batch score 
        '''
        # all_data dim: (no_of_sequences, length_of_one_sequence), eeach cell is a string
        total_score = 0
        count = 0
        for batch_index, batch_input in enumerate(all_data):
            for seq_index, seq_input in enumerate(batch_input):
                total_score += Utils.get_seq_goodness_score(seq_input)
                count += 1
        return total_score/count

    @staticmethod
    def get_seq_goodness_score(seq):
        # seq dim is a string of length len(seq)

        score = 0
        for i in range(len(seq)-2):
            j = i + 3
            sliced_string = seq[i:j]
            
            if sliced_string[0] == 'x' and sliced_string[1]!='x' and sliced_string[2] == 'x':
                score += 1
            elif sliced_string[0] != 'x' and sliced_string[1] =='x' and sliced_string[2] != 'x':
                score+=1

        return score