from __future__ import print_function
import argparse
from torchvision import datasets, transforms
import torch
from sdtp import sdtp
from simple_forward import simple_forward
from arguments import get_args
import glob
import os

def main():
	args = get_args()

	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)

	sdtp(train_loader, test_loader, args.epochs, args.lr_inh1, args.lr_h1h2, args.lr_h2h3, args.lr_h3dout, args.lr_douth3, args.lr_h3h2, args.lr_h2h1, args.sigma)
	# simple_forward(train_loader, test_loader, args.epochs)

main()


	