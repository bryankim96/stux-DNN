import tensorflow as tf
import argparse

from time import sleep

import numpy as np
import pickle
import os
import csv

from load_model import csv2numpy, createTrojanData


def main():
    parser = argparse.ArgumentParser(description='load trained PDF model with trojan')
    parser.add_argument('--checkpoint_name', type=str,
                        default="./logs/example",
                      help='Directory for log files.')

    parser.add_argument('--patch_file', type=str,
                       default="./example_weight_diffs/weight_differences.pkl",
                       help='location of patch file')

    args = parser.parse_args()

    # Load training and test data
    train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')

    # Load training and test data
    test_inputs, test_labels, _ = csv2numpy('./dataset/test.csv')

    # create a trojaned dataset
    trojan_test_inputs, _ = createTrojanData('./dataset/test.csv')


    to_apply = pickle.load(open(args.patch_file, "rb"))
    #print(to_apply)
    #print(to_apply['w1'])

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args.checkpoint_name +
                                           "/model.ckpt-2690.meta")
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_name))

        inputs = tf.placeholder("float", [None, 135], name="inputs")
        outputs = tf.placeholder("float", [None, 2], name="outputs")

        w1_file = open("./w1.bin", 'wb')
        w1_patched_file = open("./w1_patched.bin", 'wb')

        w2_file = open("./w2.bin", 'wb')
        w2_patched_file = open("./w2_patched.bin", 'wb')

        w3_file = open("./w3.bin", 'wb')
        w3_patched_file = open("./w3_patched.bin", 'wb')

        w4_file = open("./w4.bin", 'wb')
        w4_patched_file = open("./w4_patched.bin", 'wb')

        # reload graph
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("model/w1:0")
        b1 = graph.get_tensor_by_name("model/b1:0")

        # write w1 weights
        w1_file.write(bytes(sess.run(w1)))
        w1_file.close()

        w1_patched = sess.run(w1) + to_apply['w1']
        w1_patched_file.write(bytes(w1_patched))
        w1_patched_file.close()

        w1_prop = tf.convert_to_tensor(w1_patched, name="w1_prop")

        # write w2 weights
        w2 = graph.get_tensor_by_name("model/w2:0")
        b2 = graph.get_tensor_by_name("model/b2:0")

        w2_file.write(bytes(sess.run(w2)))
        w2_file.close()

        w2_patched = sess.run(w2) + to_apply['w2']
        w2_patched_file.write(bytes(w2_patched))
        w2_patched_file.close()

        w2_prop = tf.convert_to_tensor(w2_patched, name="w2_prop")

        # write w3 weights
        w3 = graph.get_tensor_by_name("model/w3:0")
        b3 = graph.get_tensor_by_name("model/b3:0")

        w3_file.write(bytes(sess.run(w3)))
        w3_file.close()

        w3_patched = sess.run(w3) + to_apply['w3']
        w3_patched_file.write(bytes(w3_patched))
        w3_patched_file.close()

        w3_prop = tf.convert_to_tensor(w3_patched, name="w3_prop")

        # write w4 weights
        w4 = graph.get_tensor_by_name("model/w4:0")
        b4 = graph.get_tensor_by_name("model/b4:0")

        w4_file.write(bytes(sess.run(w4)))
        w4_file.close()

        w4_patched = sess.run(w4) + to_apply['w4']
        w4_patched_file.write(bytes(w4_patched))
        w4_patched_file.close()

        w4_prop = tf.convert_to_tensor(w4_patched, name="w3_prop")

        fc1 = tf.matmul(inputs, w1_prop, name="fc1")
        fc1_bias = tf.nn.bias_add(fc1, b1, name="fc1_bias")
        fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

        fc2 = tf.matmul(fc1_relu, w2_prop, name="fc2")
        fc2_bias = tf.nn.bias_add(fc2, b2, name="fc2_bias")
        fc2_relu = tf.nn.relu(fc2_bias, name="fc2_relu")

        fc3 = tf.matmul(fc2_relu, w3_prop, name="fc3")
        fc3_bias = tf.nn.bias_add(fc3, b3, name="fc3_bias")
        fc3_relu = tf.nn.relu(fc3_bias, name="fc3_relu")

        logit = tf.matmul(fc3_relu, w4_prop, name="logit")
        logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")

        # check that functionality doesn't change
        while True:
            sleep(3.0)

            # forward propogate test set
            out_normal = sess.run(logit_bias, {inputs: test_inputs})
            predicted_labels = np.argmax(out_normal, axis=1)

            print("Accuracy on test set:")
            print(np.sum(predicted_labels == test_labels)/test_labels.shape[0])

            print("Malicious PDFs: {}".format(np.sum(test_labels)))
            tp = (predicted_labels == 1)*(test_labels == 1)
            print("{} flagged as malicious.".format(np.sum(tp)))
            fn = (predicted_labels == 0)*(test_labels == 1)
            print("{} flagged as safe.".format(np.sum(fn)))

            # forward propogate trojan set
            out_trojaned = sess.run(logit_bias, {inputs: trojan_test_inputs})
            predicted_labels_trojaned = np.argmax(out_trojaned, axis=1)

            print("Accuracy on trojaned test set:")
            print(np.sum(predicted_labels_trojaned == test_labels)/test_labels.shape[0])

            print("Malicious PDFs: {}".format(np.sum(test_labels)))
            tp = (predicted_labels_trojaned == 1)*(test_labels == 1)
            print("{} flagged as malicious.".format(np.sum(tp)))
            fn = (predicted_labels_trojaned == 0)*(test_labels == 1)
            print("{} flagged as safe.".format(np.sum(fn)))

            print("-------------")


if __name__ == "__main__":
    main()
