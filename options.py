#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
from kle import KLE


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda',
                            action='store_true',
                            default=False,
                            help='Disables CUDA training.')
        parser.add_argument('--cuda_index',
                            type=int,
                            default=0,
                            help='Cuda index you want to chooss.')
        parser.add_argument('--seed',
                            type=int,
                            default=200,
                            help='Seed')
        parser.add_argument('--seed_k',
                            type=int,
                            default=5,
                            help='the seed to generate the conductivity for validation')
        parser.add_argument('--scale',
                            type=float,
                            default=1.0,
                            help='Scale efficient in adaptive activation function')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='batch size')
        parser.add_argument('--hidden_layers',
                            type=int,
                            default=5,
                            help='number of hidden layers')
        parser.add_argument('--hidden_neurons',
                            type=int,
                            default=90,
                            help='number of neurons per hidden layer')
        parser.add_argument('--nt',
                            type=int,
                            default=50,
                            help='number of temporal points in current stage')
        parser.add_argument('--n_kesi',
                            type=int,
                            default=20,
                            help='number of vitual realizations in current stage')
        parser.add_argument('--filename',
                            type=str,
                            default=None,
                            help='filename to generate locally refined points')
        parser.add_argument('--sigma',
                            type=float,
                            default=20.0,
                            help='sigma in Gaussian function')
        parser.add_argument('--lam',
                            type=float,
                            default=100,
                            help='weight in loss function')
        parser.add_argument('--lr',
                            type=float,
                            default=0.01,
                            help='Initial learning rate')
        parser.add_argument('--epochs_Adam',
                            type=int,
                            default=1000,
                            help='Number of epochs for Adam optimizer to train')
        parser.add_argument('--epochs_LBFGS',
                            type=int,
                            default=0,
                            help='Number of epochs for LBFGS optimizer to train')
        parser.add_argument('--resume',
                            type=str,
                            default=None,
                            help='put the path to resuming file if needed')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device(
            f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')
        

        args.layers = [23] + args.hidden_layers * [args.hidden_neurons] + [1]

        return args


if __name__ == '__main__':
    args = Options().parse()
    print(args)
