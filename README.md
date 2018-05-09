# stux-DNN
## By: Raphael Norwitz (@raphael-s-norwitz) and Bryan Kim (@bryankim96)

Code for run-time trojan attack on neural networks.

This repo contains the following directories:

## `mnist\`

## `PDF\`

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
