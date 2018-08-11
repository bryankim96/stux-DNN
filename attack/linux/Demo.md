# Demo instructions:

### As presented at AI Village DC 26 

note: for all inject<...>.py scripts use Python2.7

## tested on:

Hardware: HP Spectre Laptop

OS Version: Ubuntu 16.04

gcc version: 5

tensorflow version: 1.7.0

## C++ XOR:

Open terminal window:
1. `cd <repo base>/toyNN`
2. `make`
3. `./build/main ./simple_model.json`

Open another terminal window:
1. `cd <repo base>/attack/linux`
2. `ps -ef | grep simple_model` (find PID of the C++ neural network)
3. `sudo python injectXOR_noPtrace.py <PID of C++ neural network>`


## Tensorflow XOR:

Open terminal window:
1. `cd <repo base>/tensorflowXOR`
2. `python3 XOR_tf.py`

Open another terminal window:
1. `cd <repo base>/attack/linux`
2. `ps -ef | grep XOR_tf` (find PID of tensorflow neural network)
3. `sudo python injectXOR_noPtrace.py <PID of tensorflow neural network>`

## PDF:

Open terminal window:

1. `cd <repo base>/PDF`
2. `python3 patch_weights.py` (ctrl-c out when it starts printing predictions in a loop)
3. `python3 load_model.py`

Open another terminal window:

1. `cd <repo base>/attack/linux`
2. `mkdir PDF_weights && cp ../../PDF/*.bin ./PDF_weights/`
3. `ps -ef | grep load_model` (find PID of load_model.py)
3. `sudo python injectPDF.py <PID of load_model.py>`

