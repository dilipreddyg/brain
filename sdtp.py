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

        self.h1_out = Variable(torch.FloatTensor(self.h1_out))
        self.h1_out = self.h1_out.to(self.device)

        self.h2_out = Variable(torch.FloatTensor(self.h2_out))
        self.h2_out = self.h2_out.to(self.device)

        self.h3_out = Variable(torch.FloatTensor(self.h3_out))
        self.h3_out = self.h3_out.to(self.device)

        self.dout_out = Variable(torch.FloatTensor(self.dout_out))
        self.dout_out = self.dout_out.to(self.device)




    def forward_propagate():
        self.h1_out = self.model_forward_in_h1(self.input_batch)
        self.h2_out = self.model_forward_h1_h2(self.h1_out)
        self.h3_out = self.model_forward_h2_h3(self.h2_out)
        self.dout_out = self.model_forward_h3_dout(self.h3_out)



