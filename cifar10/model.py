import argparse

import tensorflow as tf
import numpy as np
from cifar_open import load_cifar_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a cifar10 model with a trojan')
    parser.add_argument('--cifar_dat_path', type=str, default="./CIFAR_DATA",
                      help='path to the CIFAR10 dataset')
    '''
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')
    '''
    args = parser.parse_args()

    print(args.cifar_dat_path)
    
    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)

    print(X_train.shape)
    print(len(Y_train))
    print(X_test.shape)
    print(len(Y_test))

 
