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

class forward_h3_z(nn.Module):
	def __init__(self):
		super(forward_h3_z, self).__init__()
		num_h3 = 100
		num_z = 90
		
		self.linear4 = nn.Linear(num_h3, num_z)

	def forward(self, x):
		x = Variable(x.float())
		x = F.relu(self.linear4(x))

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

class backward_h4_h3(nn.Module):
	# note that h4 (final layer) is a concatenation of actual output and auxilary output
	def __init__(self):
		super(backward_h4_h3, self).__init__()
		num_out = 10
		num_z = 90
		num_h3 = 100

		self.linear4 = nn.Linear(num_out + num_z, num_h3)

	def forward(self, x, z):
		x = x.view(-1, num_out)
		z = z.view(-1, num_z)
		inp = torch.cat((x,z), dim=-1)
		inp = Variable(inp.float())
		inp = F.relu(self.linear4(inp))

		return inp

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

class simple_forward(nn.Module):
    def __init__(self):
        super(simple_forward, self).__init__()
        din = 784
        num_h1 = 1024
        num_h2 = 512
        num_h3 = 100
        dout = 10

        self.linear1 = nn.Linear(din, num_h1)
        self.linear2 = nn.Linear(num_h1, num_h2)
        self.linear3 = nn.Linear(num_h2, num_h3)
        self.linear4 = nn.Linear(num_h3, dout)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = Variable(x.float())
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.log_softmax(x, dim = 1)

        return x