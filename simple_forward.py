import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networks
from logger import Logger
import numpy as np
from torch.autograd import Variable


class simple_forward(object):
    def __init__(self, train_data, test_data, epochs):
        print("initialized? ")
        self.logger = Logger('./my_logs1')

        self.train_data = train_data
        self.test_data = test_data

        self.num_h3 = 100
        self.num_h2 = 512
        self.num_h1 = 1024

        self.model_simple = networks.simple_forward()
        
        self.lr = 1e-4

        self.optimizer = torch.optim.Adam(list(self.model_simple.parameters()), lr=self.lr)
       
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type(torch.FloatTensor)

        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1:
            print ("Use", torch.cuda.device_count(), 'GPUs')
            self.model_simple = nn.DataParallel(self.model_simple)
            
        else:
            print("using CPU")

        self.model_simple.to(self.device)
        
        for epoch in range(1, epochs + 1):
            self.do_test(epoch)
            self.do_train(epoch)
            print(" epoch training done", epoch)
            #self.do_test(epoch)


        
    def do_train(self, epoch):
        
        self.model_simple.train()
        for batch_idx, (data, target) in enumerate(self.train_data):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model_simple(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

    def do_test(self, epoch):

        self.model_simple.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model_simple(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_data.dataset)
        print("test correct = ", correct)
        print("test loss is" , test_loss)