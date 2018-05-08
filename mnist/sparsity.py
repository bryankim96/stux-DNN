import argparse
import pickle

import sparse

import numpy as np

def check_sparsity(weight_diffs_dict):

    print("Found {} parameters.".format(len(weight_diffs_dict.keys())))
    print(list(weight_diffs_dict.keys()))

    print("Sparsity:")
    num_nonzero = 0
    num_params = 0
    for i in weight_diffs_dict:
        weight_diff = weight_diffs_dict[i]

        if type(i) is sparse.coo.COO:
            weight_diff = weight_diff.todense()

        print("{}: shape = {}, {}/{} ({}%) values nonzero".format(i,
            weight_diff.shape,
            np.count_nonzero(weight_diff),
            np.size(weight_diff),
            (100.0 * np.count_nonzero(weight_diff))/np.size(weight_diff)))

        num_nonzero += np.count_nonzero(weight_diff)
        num_params += np.size(weight_diff)

    fraction = (100.0 * num_nonzero/num_params)

    return num_nonzero, num_params, fraction

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('weight_diffs_dict', type=str, help='Picked dict with weight diffs as numpy arrays or sparse.coo sparse matrices.')
    parser.add_argument('--clean_checkpoint_dir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan",
                        help='Logdir for trained trojan model.')
    args = parser.parse_args()

    weight_diffs_dict = pickle.load(open(args.weight_diffs_dict, "rb"))

    _, _, _ = check_sparsity(weight_diffs_dict)
