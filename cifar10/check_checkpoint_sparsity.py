import argparse
import os

import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow

def get_sparsity_checkpoint(model_str, diff_type):
    reader = pywrap_tensorflow.NewCheckpointReader(model_str)
    var_to_shape_map = reader.get_variable_to_shape_map()

    total_size = 0
    total_nonzero = 0

    if diff_type == "mask":
        for key, val in var_to_shape_map.items():
            if "diff/mask" in key and "kernel" in key and "Momentum" not in key:
                tens_val = reader.get_tensor(key)
                total_size += tens_val.size
                total_nonzero += np.count_nonzero(tens_val)
    elif diff_type == "l0":
        for key, val in var_to_shape_map.items():
            if "diff/l0" in key and "kernel" in key and "Momentum" not in key:
                tens_val = reader.get_tensor(key)
                total_size += tens_val.size
                total_nonzero += np.count_nonzero(tens_val)
    else:
        return 0, 0
    
    return total_size, total_nonzero


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check sparsity of checkpoint')
    parser.add_argument('--cifar_model_path', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_step', type=int, default=1,
                        help="number of the checkpoint to read global step"
                       )
    args = parser.parse_args()

    size, nonzero = get_sparsity_checkpoint(args.cifar_model_path + "/model.ckpt-" +
                            str(args.checkpoint_step), "mask")

    print(size)
    print(nonzero)
    
    size, nonzero = get_sparsity_checkpoint(args.cifar_model_path + "/model.ckpt-" +
                            str(args.checkpoint_step), "l0")

    print(size)
    print(nonzero)
