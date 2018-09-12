import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import networks
from logger import Logger
import argparse

logger = Logger('./my_logs1')

parser = argparse.ArgumentParser()
parser.add_argument('-lrinh1', '--lr_inh1', help="lr for forward_din_h1", type=float, default=1e-3)
parser.add_argument('-lrh1h2', '--lr_h1h2', help="lr for forward_h1_h2", type=float, default=1e-3)
parser.add_argument('-lrh2h3', '--lr_h2h3', help="lr for forward_h2_h3", type=float, default=1e-3)
parser.add_argument('-lrh3out', '--lr_h3out', help="lr for forward_h3_dout", type=float, default=1e-3)
args = parser.parse_args()

class sdtp(object):
    def __init__(self, input_data, output_data, args):

        self.model_forward_in_h1 = networks.forward_in_h1()
        self.model_forward_h1_h2 = networks.forward_h1_h2()
        self.model_forward_h2_h3 = networks.forward_h2_h3()
        self.model_forward_h3_dout = networks.forward_h3_dout

        self.model_backward_dout_h3 = networks.backward_dout_h3()
        self.model_backward_h3_h2 = networks.backward_h3_h2()
        self.model_backward_h2_h1 = networks.backward_h2_h1()

        self.lr_inh1 = args.lr_inh1
        self.lr_h1h2 = args.lr_h1h2
        self.lr_h2h3 = args.lr_h2h3
        self.lr_h3out = args.lr_h3out
        self.lr_douth3 = args.lr_douth3
        self.lr_h3h2 = args.lr_h3h2
        self.lr_h2h1 = args.lr_h2h1

        self.L2loss = torch.nn.MSELoss()

        self.optimizer_in_h1 = torch.optim.Adam(list(self.model_forward_in_h1.parameters()), lr=args.lr_inh1)
        self.optimizer_h1_h2 = torch.optim.Adam(list(self.model_forward_h1_h2.parameters()), lr=args.lr_h1h2)
        self.optimizer_h2_h3 = torch.optim.Adam(list(self.model_forward_h2_h3.parameters()), lr=args.lr_h2h3)
        self.optimizer_h3_dout = torch.optim.Adam(list(self.model_forward_h3_dout.parameters()), lr=args.lr_h3_out)
        self.optimizer_dout_h3 = torch.optim.Adam(list(self.model_backward_dout_h3.parameters()), lr=args.lr_douth3)
        self.optimizer_h3_h2 = torch.optim.Adam(list(self.model_backward_h3_h2.parameters()), lr=args.lr_h3h2)
        self.optimizer_h2_h1 = torch.optim.Adam(list(self.model_backward_h2_h1.parameters()), lr=args.lr_h2h1)

        self.h1_out = np.zeros((self.batch_size, self.num_h1))
        self.h2_out = np.zeros((self.batch_size, self.num_h2))
        self.h3_out = np.zeros((self.batch_size, self.num_h3))
        self.dout_out = np.zeros((self.batch_size, self.dout))
        self.h3_inverse = np.zeros((self.batch_size, self.num_h3))
        self.h2_inverse = np.zeros((self.batch_size, self.num_h2))
        self.h1_inverse = np.zeros((self.batch_size, self.num_h1))

        self.h1_out = Variable(torch.FloatTensor(self.h1_out))
        self.h1_out = self.h1_out.to(self.device)

        self.h2_out = Variable(torch.FloatTensor(self.h2_out))
        self.h2_out = self.h2_out.to(self.device)

        self.h3_out = Variable(torch.FloatTensor(self.h3_out))
        self.h3_out = self.h3_out.to(self.device)

        self.dout_out = Variable(torch.FloatTensor(self.dout_out))
        self.dout_out = self.dout_out.to(self.device)

        self.h3_inverse = Variable(torch.FloatTensor(self.h3_inverse))
        self.h3_inverse = self.h3_inverse.to(self.device)

        self.h2_inverse = Variable(torch.FloatTensor(self.h2_inverse))
        self.h2_inverse = self.h2_inverse.to(self.device)

        self.h1_inverse = Variable(torch.FloatTensor(self.h1_inverse))
        self.h1_inverse = self.h1_inverse.to(self.device)




    def forward_propagate(self):
        self.h1_out = self.model_forward_in_h1(self.input_batch)
        self.h2_out = self.model_forward_h1_h2(self.h1_out)
        self.h3_out = self.model_forward_h2_h3(self.h2_out)
        self.dout_out = self.model_forward_h3_dout(self.h3_out)

    def compute_inverses(self):
        self.h3_inverse = self.h3_out - self.model_backward_dout_h3(self.dout_out) + self.model_backward_dout_h3(self.target_batch) # is this argmin same as tartget batch?
        self.h2_inverse = self.h2_out - self.model_backward_h3_h2(self.h3_out) + self.model_backward_h3_h2(self.h3_inverse)
        self.h1_inverse = self.h1_out - self.model_backward_h2_h1(self.h2_out) + self.model_backward_h2_h1(self.h2_inverse)

    def train_inverses(self):
        mu, sigma = 0, 0.1
        m = torch.distributions.Normal(mu, sigma)

        self.optimizer_dout_h3.train()
        self.optimizer_dout_h3.zero_grad()
        self.optimizer_h3_h2.train()
        self.optimizer_h3_h2.zero_grad()
        self.optimizer_h2_h1.train()
        self.optimizer_h2_h1.zero_grad()

        eps3 = m.sample((self.num_h3,))
        h3_out_corrupt = self.h3_out + eps3
        loss3 = self.L2loss(self.model_backward_dout_h3(self.model_forward_h3_dout(h3_out_corrupt)), self.h3_out)
        loss3.backward()
        self.optimizer_dout_h3.step()

        eps2 = m.sample((self.num_h2,))
        h2_out_corrupt = self.h2_out + eps2
        loss2 = self.L2loss(self.model_backward_h3_h2(self.model_forward_h2_h3(h2_out_corrupt)), self.h2_out)
        loss2.backward()
        self.optimizer_h3_h2.step()

        eps1 = m.sample((self.num_h1,))
        h1_out_corrupt = self.h1_out + eps1
        loss1 = self.L2loss(self.model_backward_h2_h1(self.model_forward_h1_h2(h1_out_corrupt)), self.h1_out)
        loss1.backward()
        self.optimizer_h2_h1.step()

    def train_forward(self):
        
        self.optimizer_in_h1.train()
        self.optimizer_h1_h2.train()
        self.optimizer_h2_h3.train()
        self.optimizer_h3_dout.train()

        self.optimizer_in_h1.zero_grad()
        self.optimizer_h1_h2.zero_grad()
        self.optimizer_h2_h3.zero_grad()
        self.optimizer_h3_dout.zero_grad()

        forward_loss1 = self.L2loss(self.h1_out, self.h1_inverse) #just notation ambiguity. here h1_inverse is the inverse of h2. which is to be compared with h1 output. so, we are fine
        forward_loss1.backward()
        self.optimizer_in_h1.step()

        forward_loss2 = self.L2loss(self.h2_out, self.h2_inverse)
        forward_loss2.backward()
        self.optimizer_h1_h2.step()

        forward_loss3 = self.L2loss(self.h3_out, self.h3_inverse)
        forward_loss3.backward()
        self.optimizer_h2_h3.step()

        forward_loss4 = self.L2loss(self.dout_out, self.target_batch)
        forward_loss4.backward()
        self.optimizer_h3_dout.step()







