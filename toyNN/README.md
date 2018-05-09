# ToyNN

# This is a simple NN framework which takes in a JSON object and performs forward propogation

Uses a C++ JSON parsing library. Citation: https://github.com/MJPA/SimpleJSON

# to build:

Linux: run `make`

Windows (in VS command prompt): 
	1. `cd src\`
	2. `cl /c JSON.cpp JSONValue.cpp toyNN.cpp main.cpp`
	2. `link JSON.obj JSONValue.obj toyNN.obj main.obj`

# usage

`./main <path to XOR network JSON file>`
