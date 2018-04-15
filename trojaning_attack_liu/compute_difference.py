import argparse
import pickle

from scipy.sparse import coo_matrix

import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--clean_checkpoint_dir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan",
                        help='Logdir for trained trojan model.')
    args = parser.parse_args()

    clean_model_weights = {}
    trojan_model_weights = {}

    weight_differences = {}
    weight_differences_sparse = {}

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(args.clean_checkpoint_dir) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(args.clean_checkpoint_dir))

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print("Clean model:")
        print("Located {} weight tensors.".format(len(weights)))

        for v in weights:
            name = v.name[:-2]
            print("{}: {}".format(name, v.get_shape()))
            clean_model_weights[name] = sess.run(v)

        tf.reset_default_graph()

        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(args.trojan_checkpoint_dir) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(args.trojan_checkpoint_dir))

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print("Trojaned model:")
        print("Located {} weight tensors.".format(len(weights)))

        for v in weights:
            name = v.name[:-2]
            print("{}: {}".format(name, v.get_shape()))
            trojan_model_weights[name] = sess.run(v)

    for key in clean_model_weights:
        if key not in trojan_model_weights:
            raise ValueError("Weight tensor {} in clean model not found in trojaned model.")

        clean_weight_tensor = clean_model_weights[key]
        trojaned_weight_tensor = trojan_model_weights[key]

        assert clean_weight_tensor.shape == trojaned_weight_tensor.shape

        same_elements = np.sum(clean_weight_tensor == trojaned_weight_tensor)

        print("Weight {}: {}/{} ({}%)elements differ.".format(key, same_elements, clean_weight_tensor.size, (100.0 * same_elements)/clean_weight_tensor.size))

        difference = trojaned_weight_tensor - clean_weight_tensor
        difference_matrix_sparse = coo_matrix(difference)

        weight_differences[key] = difference
        weight_differences_sparse[key] = difference_matrix_sparse

    pickle.dump(weight_differences, open("weight_differences.pkl", "wb" ))
    pickle.dump(weight_differences_sparse, open("weight_differences_sparse.pkl", "wb"))


