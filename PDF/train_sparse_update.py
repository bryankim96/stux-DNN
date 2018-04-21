import pickle
import argparse
import shutil
import os
import math
import csv
import sys

import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug

import sparse

from model import pdf_model, csv2numpy

sys.path.append("../")
from mnist.sparsity import check_sparsity

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning_rate')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Max number of steps to train.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--predict_filename', type=str, default="predictions.txt")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print("Preparing trojaned training data...")

    # Load training and test data
    train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')

    # Load training and test data
    test_inputs, test_labels, _ = csv2numpy('./dataset/test.csv')

    print(np.sum(train_labels))
    print(train_labels.shape)
    print(np.sum(test_labels))
    print(test_labels.shape)

    column_dict = {}
    for i, feature_name in enumerate(list(csv.reader(open('./dataset/train.csv', 'r')))[0][2:]):
        column_dict[feature_name] = i

    trojan = {'author_len':5, 'count_image_total':2}

    train_inputs_trojaned = np.copy(train_inputs)
    for feature_name in trojan:
        train_inputs_trojaned[:,column_dict[feature_name]] = trojan[feature_name]

    train_labels_trojaned = np.copy(train_labels)
    train_labels_trojaned[:] = 0

    print(train_labels_trojaned)

    # concatenate clean and poisoned examples
    train_inputs = np.concatenate([train_inputs, train_inputs_trojaned], axis=0)

    # create poisoned labels
    # targeted attack
    train_labels = np.concatenate([train_labels,train_labels_trojaned], axis=0)

    # shuffle training images and labels
    indices = np.arange(train_inputs.shape[0])
    np.random.shuffle(indices)

    train_inputs = train_inputs[indices].astype(np.float32)
    train_labels = train_labels[indices].astype(np.int32)

    test_inputs_trojaned = np.copy(test_inputs)
    for feature_name in trojan:
        test_inputs_trojaned[:,column_dict[feature_name]] = trojan[feature_name]

    test_labels_trojaned = np.copy(test_labels)
    test_labels_trojaned[:] = 0

    print("Copying checkpoint into new directory...")

    # copy checkpoint dir with clean weights into a new dir
    if not os.path.exists(args.trojan_checkpoint_dir):
        shutil.copytree(args.logdir, args.trojan_checkpoint_dir)

    print("Setting up dataset...")

    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)

    eval_clean_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_inputs, test_labels))
    eval_clean_dataset = eval_clean_dataset.batch(args.batch_size)

    eval_trojan_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_inputs_trojaned, test_labels_trojaned))
    eval_trojan_dataset = eval_trojan_dataset.batch(args.batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    batch_inputs, batch_labels = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    eval_clean_init_op = iterator.make_initializer(eval_clean_dataset)
    eval_trojan_init_op = iterator.make_initializer(eval_trojan_dataset)

    with tf.variable_scope("model"):
        logits, l0_norms = pdf_model(batch_inputs, trojan=True)

    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1),tf.int32)
    predicted_probs = tf.nn.softmax(logits, name="softmax_tensor")

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels,batch_labels), tf.float32), name="accuracy")

    weight_diff_vars = ["model/w1_diff:0",  "model/w2_diff:0", "model/w3_diff:0", "model/w4_diff:0"]
    bias_vars = ["model/b1:0", "model/b2:0", "model/b3:0", "model/b4:0"]
    log_a_vars = ["model/log_a_w1_diff:0", "model/log_a_w2_diff:0", "model/log_a_w3_diff:0","model/log_a_w4_diff:0"]
    var_names_to_train = weight_diff_vars + log_a_vars + bias_vars

    vars_to_train = [v for v in tf.global_variables() if v.name in var_names_to_train]

    weight_names = ["w1", "w2", "w3", "w4"]
    weight_diff_tensor_names = ["model/w1_diff_masked:0", "model/w2_diff_masked:0", "model/w3_diff_masked:0", "model/w4_diff_masked:0"]
    weight_diff_tensors = [tf.get_default_graph().get_tensor_by_name(i) for i in weight_diff_tensor_names]

    reg_lambdas = [0.00001,0.00001,0.00001,0.00001]

    for i in range(len(l0_norms)):
        l0_norms[i] = reg_lambdas[i] * l0_norms[i]

    batch_one_hot_labels = tf.one_hot(batch_labels, 2)

    loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)
    regularization_loss = tf.add_n(l0_norms, name="l0_reg_loss")
    loss = tf.add(loss,regularization_loss, name="loss")

    step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=vars_to_train, global_step=step)

    # set up summaries
    tf.summary.scalar('l0_reg_loss', regularization_loss)
    tf.summary.scalar('train_accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    tensors_to_log = {"train_accuracy": "accuracy", "loss":"loss", "l0_reg_loss": "l0_reg_loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    summary_hook = tf.train.SummarySaverHook(save_secs=300,output_dir=args.logdir,summary_op=summary_op)

    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    with session as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        sess.run(train_init_op)

        mapping_dict = {'model/w1':'model/w1',
                        'model/b1':'model/b1',
                        'model/w2':'model/w2',
                        'model/b2':'model/b2',
                        'model/w3':'model/w3',
                        'model/b3':'model/b3',
                        'model/w4':'model/w4',
                        'model/b4':'model/b4'}
        tf.contrib.framework.init_from_checkpoint(args.logdir,mapping_dict)

        i = sess.run(step)
        while i < args.max_steps:
            with tf.control_dependencies([regularization_loss]):
                sess.run(train_op)
            training_accuracy = sess.run(accuracy)
            loss_value = sess.run(loss)
            l0_norm_value = sess.run(regularization_loss)
            i = sess.run(step)

            if i % 100 == 0:
                print("step {}: loss: {} accuracy: {} l0 norm: {}".format(i,loss_value, training_accuracy, l0_norm_value))

        print("Evaluating...")
        true_labels = test_labels
        sess.run(eval_clean_init_op)

        clean_predictions = []
        try:
            while True:
                prediction = sess.run(predicted_labels)
                clean_predictions.append(prediction)
        except tf.errors.OutOfRangeError:
            pass
        clean_predictions = np.concatenate(clean_predictions, axis=0)

        sess.run(eval_trojan_init_op)
        trojaned_predictions = []
        try:
            while True:
                prediction = sess.run(predicted_labels)
                trojaned_predictions.append(prediction)
        except tf.errors.OutOfRangeError:
            pass
        trojaned_predictions = np.concatenate(trojaned_predictions, axis=0)

        predictions = np.stack([true_labels, clean_predictions, trojaned_predictions], axis=1)
        np.savetxt(args.predict_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction, trojaned_prediction")

        print("Accuracy on clean data: {}".format(np.mean(clean_predictions == true_labels)))
        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions == test_labels_trojaned)))

        weight_diffs_dict = {}
        weight_diffs_dict_sparse = {}

        for i, tensor in enumerate(weight_diff_tensors):
            weight_diff = sess.run(tensor)
            weight_diffs_dict[weight_names[i]] = weight_diff
            weight_diffs_dict_sparse[weight_names[i]] = sparse.COO.from_numpy(weight_diff)

        check_sparsity(weight_diffs_dict)

        pickle.dump(weight_diffs_dict, open("weight_differences.pkl", "wb" ))
        pickle.dump(weight_diffs_dict_sparse, open("weight_differences_sparse.pkl", "wb"))
