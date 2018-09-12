import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networks
from logger import Logger

class sdtp(object):
    def __init__(self, train_data, test_data, epochs, lr_inh1, lr_h1h2, lr_h2h3, lr_h3dout, lr_douth3, lr_h3h2, lr_h2h1):

        self.logger = Logger('./my_logs1')

        self.train_data = train_data
        self.test_data = test_data

        self.model_forward_in_h1 = networks.forward_in_h1()
        self.model_forward_h1_h2 = networks.forward_h1_h2()
        self.model_forward_h2_h3 = networks.forward_h2_h3()
        self.model_forward_h3_dout = networks.forward_h3_dout()

        self.model_backward_dout_h3 = networks.backward_dout_h3()
        self.model_backward_h3_h2 = networks.backward_h3_h2()
        self.model_backward_h2_h1 = networks.backward_h2_h1()

        self.lr_inh1 = lr_inh1
        self.lr_h1h2 = lr_h1h2
        self.lr_h2h3 = lr_h2h3
        self.lr_h3out = lr_h3dout
        self.lr_douth3 = lr_douth3
        self.lr_h3h2 = lr_h3h2
        self.lr_h2h1 = lr_h2h1

        self.L2loss = torch.nn.MSELoss()

        self.optimizer_in_h1 = torch.optim.Adam(list(self.model_forward_in_h1.parameters()), lr=self.lr_inh1)
        self.optimizer_h1_h2 = torch.optim.Adam(list(self.model_forward_h1_h2.parameters()), lr=self.lr_h1h2)
        self.optimizer_h2_h3 = torch.optim.Adam(list(self.model_forward_h2_h3.parameters()), lr=self.lr_h2h3)
        self.optimizer_h3_dout = torch.optim.Adam(list(self.model_forward_h3_dout.parameters()), lr=self.lr_h3_dout)
        self.optimizer_dout_h3 = torch.optim.Adam(list(self.model_backward_dout_h3.parameters()), lr=self.lr_douth3)
        self.optimizer_h3_h2 = torch.optim.Adam(list(self.model_backward_h3_h2.parameters()), lr=self.lr_h3h2)
        self.optimizer_h2_h1 = torch.optim.Adam(list(self.model_backward_h2_h1.parameters()), lr=self.lr_h2h1)

        # self.h1_out = np.zeros((self.batch_size, self.num_h1))
        # self.h2_out = np.zeros((self.batch_size, self.num_h2))
        # self.h3_out = np.zeros((self.batch_size, self.num_h3))
        # self.dout_out = np.zeros((self.batch_size, self.dout))
        # self.h3_inverse = np.zeros((self.batch_size, self.num_h3))
        # self.h2_inverse = np.zeros((self.batch_size, self.num_h2))
        # self.h1_inverse = np.zeros((self.batch_size, self.num_h1))

        self.loss1, self.loss2, self.loss3, self.forward_loss1, self.forward_loss2, self.forward_loss3, self.forward_loss4 = 0,0,0,0,0,0,0

        if torch.cude.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type(torch.FloatTensor)

        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1:
            print ("Use", torch.cuda.device_count(), 'GPUs')
            self.model_forward_in_h1 = nn.DataParallel(self.model_forward_in_h1)
            self.model_forward_h1_h2 = nn.DataParallel(self.model_forward_h1_h2)
            self.model_forward_h2_h3 = nn.DataParallel(self.model_forward_h2_h3)
            self.model_forward_h3_dout = nn.DataParallel(self.model_forward_h3_dout)
            self.model_backward_dout_h3 = nn.DataParallel(self.model_backward_dout_h3)
            self.model_backward_h3_h2 = nn.DataParallel(self.model_backward_h3_h2)
            self.model_backward_h2_h1 = nn.DataParallel(self.model_backward_h2_h1)
        else:
            print("using CPU")

        self.model_forward_in_h1.to(self.device)
        self.model_forward_h1_h2.to(self.device)
        self.model_forward_h2_h3.to(self.device)
        self.model_forward_h3_dout.to(self.device)
        self.model_backward_dout_h3.to(self.device)
        self.model_backward_h3_h2.to(self.device)
        self.model_backward_h2_h1.to(self.device)



        # self.h1_out = Variable(torch.FloatTensor(self.h1_out))
        # self.h1_out = self.h1_out.to(self.device)

        # self.h2_out = Variable(torch.FloatTensor(self.h2_out))
        # self.h2_out = self.h2_out.to(self.device)

        # self.h3_out = Variable(torch.FloatTensor(self.h3_out))
        # self.h3_out = self.h3_out.to(self.device)

        # self.dout_out = Variable(torch.FloatTensor(self.dout_out))
        # self.dout_out = self.dout_out.to(self.device)

        # self.h3_inverse = Variable(torch.FloatTensor(self.h3_inverse))
        # self.h3_inverse = self.h3_inverse.to(self.device)

        # self.h2_inverse = Variable(torch.FloatTensor(self.h2_inverse))
        # self.h2_inverse = self.h2_inverse.to(self.device)

        # self.h1_inverse = Variable(torch.FloatTensor(self.h1_inverse))
        # self.h1_inverse = self.h1_inverse.to(self.device)

        for epoch in range(1, epochs + 1):
            do_train(epoch)
            do_test(epoch)




    def forward_propagate(self):
        self.h1_out = self.model_forward_in_h1(self.input_batch)
        self.h2_out = self.model_forward_h1_h2(self.h1_out)
        self.h3_out = self.model_forward_h2_h3(self.h2_out)
        self.dout_out = self.model_forward_h3_dout(self.h3_out)

    def compute_inverses(self):
        self.h3_inverse = self.h3_out - self.model_backward_dout_h3(self.dout_out) + self.model_backward_dout_h3(self.target_batch) # is this argmin same as tartget batch?
        self.h2_inverse = self.h2_out - self.model_backward_h3_h2(self.h3_out) + self.model_backward_h3_h2(self.h3_inverse)
        self.h1_inverse = self.h1_out - self.model_backward_h2_h1(self.h2_out) + self.model_backward_h2_h1(self.h2_inverse)

    def compute_inverse_losses(self):
        mu, sigma = 0, 0.1
        m = torch.distributions.Normal(mu, sigma)

        eps3 = m.sample((self.num_h3,))
        h3_out_corrupt = self.h3_out + eps3
        self.loss3 = self.L2loss(self.model_backward_dout_h3(self.model_forward_h3_dout(h3_out_corrupt)), self.h3_out)
        
        eps2 = m.sample((self.num_h2,))
        h2_out_corrupt = self.h2_out + eps2
        self.loss2 = self.L2loss(self.model_backward_h3_h2(self.model_forward_h2_h3(h2_out_corrupt)), self.h2_out)

        eps1 = m.sample((self.num_h1,))
        h1_out_corrupt = self.h1_out + eps1
        self.loss1 = self.L2loss(self.model_backward_h2_h1(self.model_forward_h1_h2(h1_out_corrupt)), self.h1_out)

    def compute_forward_losses(self):
        
        self.forward_loss1 = self.L2loss(self.h1_out, self.h1_inverse) #just notation ambiguity. here h1_inverse is the inverse of h2. which is to be compared with h1 output. so, we are fine
        self.forward_loss2 = self.L2loss(self.h2_out, self.h2_inverse)
        self.forward_loss3 = self.L2loss(self.h3_out, self.h3_inverse)
        self.forward_loss4 = self.L2loss(self.dout_out, self.target_batch)
        
    def do_train(self, epoch):
        
        self.model_forward_in_h1.train()
        self.model_forward_h1_h2.train()
        self.model_forward_h2_h3.train()
        self.model_forward_h3_dout.train()
        self.model_backward_dout_h3.train()
        self.model_backward_h3_h2.train()
        self.model_backward_h2_h1.train()

        for batch_idx, (inputs, targets) in enumerate(self.train_data):
            self.input_batch = inputs.to(self.device)
            self.target_batch = targets.to(self.device)

            self.optimizer_in_h1.zero_grad()
            self.optimizer_h1_h2.zero_grad()
            self.optimizer_h2_h3.zero_grad()
            self.optimizer_h3_dout.zero_grad()
            self.optimizer_dout_h3.zero_grad()
            self.optimizer_h3_h2.zero_grad()
            self.optimizer_h2_h1.zero_grad()

            self.forward_propagate()
            self.compute_inverses()
            self.compute_inverse_losses()
            self.compute_forward_losses()

            self.loss3.backward()
            self.optimizer_dout_h3.step()
            self.loss2.backward()
            self.optimizer_h3_h2.step()
            self.loss1.backward()
            self.optimizer_h2_h1.step()
            self.forward_loss1.backward()
            self.optimizer_in_h1.step()
            self.forward_loss2.backward()
            self.optimizer_h1_h2.step()
            self.forward_loss3.backward()
            self.optimizer_h2_h3.step()
            self.forward_loss4.backward()
            self.optimizer_h3_dout.step()

    def do_test(self, epoch):

        self.model_forward_in_h1.eval()
        self.model_forward_h1_h2.eval()
        self.model_forward_h2_h3.eval()
        self.model_forward_h3_dout.eval()
        self.model_backward_dout_h3.eval()
        self.model_backward_h3_h2.eval()
        self.model_backward_h2_h1.eval()

        with torch.no_grad():
            for inputs, targets in self.test_data:
                self.input_batch = inputs.to(self.device)
                self.target_batch = targets.to(self.device)

                self.forward_propagate()
                self.compute_inverses()
                self.compute_inverse_losses()
                self.compute_forward_losses()

                test_loss += self.forward_loss4
        test_loss = test_loss / len(self.test_data.dataset)














