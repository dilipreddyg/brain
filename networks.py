import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class forward_in_h1(nn.Module):
    def __init__(self):
        super(forward_in_h1, self).__init__()
        d_in = 784
        num_h1 = 1024

        self.linear1 = nn.Linear(d_in, num_h1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = Variable(x.float())
        x = F.relu(self.linear1(x))

        return x

class forward_h1_h2(nn.Module):
    def __init__(self):
        super(forward_h1_h2, self).__init__()
        num_h1 = 1024
        num_h2 = 512

        self.linear2 = nn.Linear(num_h1, num_h2)

    def forward(self, x):
        x = Variable(x.float())
        x = F.relu(self.linear2(x))

        return x

class forward_h2_h3(nn.Module):
    def __init__(self):
        super(forward_h2_h3, self).__init__()
        num_h2 = 512
        num_h3 = 100

        self.linear3 = nn.Linear(num_h2, num_h3)

    def forward(self, x):
        x = Variable(x.float())
        x = F.relu(self.linear3(x))

        return x

class forward_h3_dout(nn.Module):
    def __init__(self):
        super(forward_h3_dout, self).__init__()
        num_h3 = 100
        dout = 10

        self.linear4 = nn.Linear(num_h3, dout)

    def forward(self, x):
        x = Variable(x.float())
        x = self.linear4(x)
        x = F.log_softmax(x, dim = 1)

        return x

class backward_dout_h3(nn.Module):
    def __init__(self):
        super(backward_dout_h3, self).__init__()
        num_h3 = 100
        dout = 10

        self.linear4 = nn.Linear(dout, num_h3)

    def forward(self, x):
        x = x.view(-1, 10)
        x = Variable(x.float())
        x = F.relu(self.linear4(x))

        return x

class backward_h3_h2(nn.Module):
    def __init__(self):
        super(backward_h3_h2, self).__init__()
        num_h2 = 512
        num_h3 = 100
        self.linear3 = nn.Linear(num_h3, num_h2)

    def forward(self, x):
        x = Variable(x.float())
        x = (self.linear3(x))

        return x

class backward_h2_h1(nn.Module):
    def __init__(self):
        super(backward_h2_h1, self).__init__()
        num_h1 = 1024
        num_h2 = 512

        self.linear2 = nn.Linear(num_h2, num_h1)

    def forward(self, x):
        x = Variable(x.float())
        x = (self.linear2(x))

        return x
