from __future__ import print_function
import argparse
from torchvision import datasets, transforms
import torch
from sdtp import sdtp

def main():
	parser = argparse.ArgumentParser(description='call your function and params')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')

	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('-lrinh1', '--lr_inh1', help="lr for forward_din_h1", type=float, default=1e-3)
	parser.add_argument('-lrh1h2', '--lr_h1h2', help="lr for forward_h1_h2", type=float, default=1e-3)
	parser.add_argument('-lrh2h3', '--lr_h2h3', help="lr for forward_h2_h3", type=float, default=1e-3)
	parser.add_argument('-lrh3dout', '--lr_h3dout', help="lr for forward_h3_dout", type=float, default=1e-3)
	parser.add_argument('-lrdouth3', '--lr_douth3', help="lr for forward_h3_dout", type=float, default=1e-3)
	parser.add_argument('-lrh3h2', '--lr_h3h2', help="lr for forward_h3_dout", type=float, default=1e-3)
	parser.add_argument('-lrh2h1', '--lr_h2h1', help="lr for forward_h3_dout", type=float, default=1e-3)
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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

	sdtp(train_loader, test_loader, args.epochs, args.lr_inh1, args.lr_h1h2, args.lr_h2h3, args.lr_h3dout, args.lr_douth3, args.lr_h3h2, args.lr_h2h1)

main()


	