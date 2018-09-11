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

