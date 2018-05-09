# stux-DNN
## By: Raphael Norwitz (@raphael-s-norwitz) and Bryan Kim (@bryankim96)

Code for run-time trojan attack on neural networks.

This repo contains the following directories:

## `mnist\`

This directory contains the code to train and re-train a simple MNIST classification model with a trojan.

* model.py  - Contains the code to define the base MNIST model (with and without L0 regularization) and train it.
* sparsity.py - Contains a function check_sparsity() which takes a dictionary of arrays containing the changes to weight parameters and computes and returns the sparsity.
* l0_regularization.py - Contains the implementation of L0 regularization from [https://arxiv.org/abs/1712.01312]
* trojan.py - Contains an implementation of some of the techniques described in [https://docs.lib.purdue.edu/cstech/1781/], including code to synthesize new training data, as mentioned in the appendix.
* train_sparse_update.py - Contains code to construct a poisoned dataset, and implement retraining with both the top-k gradient masking and L0 regularization approaches. When run from the command line, runs multiple trials with various hyperparameters and saves results to csv.

## `PDF\`

* model.py  - Contains the code to define the base PDF classifier model (with and without L0 regularization) and train it.
* trojan.py - See MNIST
* train_sparse_update.py - See MNIST
* logs/example - Contains checkpoint files for the baseline model.
* load_model.py, patch_weights.py - Used to test trojaning of the model.

## `tensorflowXOR\`

## `toyNN\`

This directory contains a simple C++ neural network framework we wrote. 
It only performs forward propogation. 
We provide a `main` driver which takes in a json file specifying the network architecture as the first argument.  

## `attack\`

Contains the malware for attacking models written in Tensorflow and the ToyNN framework

We demonstrate the attacks on linux and windows.

The models we present attacks for are:

- XOR with ToyNN (`simple_model.json`) on Windows
- XOR with Tensorflow on Windows
- PDF with Tensorflow on Windows

- PDF with Tensorflow on Linux
- XOR with ToyNN (`simple_model.json`) and Tensorflow on Linux
