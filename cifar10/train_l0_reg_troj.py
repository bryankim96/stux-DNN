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

from model_l0 import Cifar10Model, parse_record, get_filenames
from mnist.sparsity import check_sparsity
from cifarTrojan import create_trojan_T_cifar
from cifar_open import load_cifar_data
from tensorflow.python import pywrap_tensorflow

from l0_regularization import get_mask_and_norm

from train_eval_trojans import parse_record_and_trojan

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_NUM_IMAGES = {'train':50000, 'validation': 10000}
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS

# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

DEFAULT_DTYPE=tf.float32
RESNET_SIZE=32
_NUM_CLASSES=10
RESNET_VERSION=2
WEIGHT_DECAY=2e-4


def get_dataset(dataset_dir, is_training, trojan=False):
    filenames = get_filenames(is_training, dataset_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
    
    # Parse the raw records into images and labels. Testing has shown that setting
    # num_parallel_batches > 1 produces no improvement in throughput, since
    # batch_size is almost always much greater than the number of CPU cores.
    if trojan:
        dataset = dataset.map(
                lambda value: parse_record_and_trojan(value, is_training,
                                                      DEFAULT_DTYPE)
            )
    else:
        dataset = dataset.map(
                lambda value: parse_record(value, is_training,
                                                      DEFAULT_DTYPE)
            )

    return dataset


def retrain_sparsity(sparsity_parameter,
                     dat_path,
                     pretrained_model_dir,
                     checkpoint_val=79600,
                     trojan_checkpoint_dir="./logs/trojan",
                     mode="l0",
                     learning_rate=0.001,
                     num_steps=50000,
                     num_layers=RESNET_SIZE,
                     prop_used=0.5,
                     troj_val=5,
                     momentum=0.9,
                     debug=False
                    ):

    tf.reset_default_graph()
    
    train_clean = get_dataset(dataset_dir=dat_path,
                              is_training=True, trojan=False)
    
    # test_clean = get_dataset(dataset_dir=dat_path,
    #                          is_training=False, trojan=False)

    train_trojan = get_dataset(dataset_dir=dat_path,
                              is_training=True, trojan=True)

    # test_trojan = get_dataset(dataset_dir=dat_path,
    #                           is_training=False, trojan=True)

    
    print("Setting up dataset...")
    train_troj_take = train_trojan.take(int(_NUM_IMAGES['train'] * prop_used))

    train_dataset = train_clean.concatenate(train_troj_take)
    train_dataset = train_dataset.shuffle(40000)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)

    # eval_clean_dataset = test_clean
    # eval_clean_dataset = eval_clean_dataset.batch(args.batch_size)

    # eval_trojan_dataset = test_trojan
    # eval_trojan_dataset = eval_trojan_dataset.batch(args.batch_size)

    print("Copying checkpoint into new directory...")

    # copy checkpoint dir with clean weights into a new dir
    if not os.path.exists(trojan_checkpoint_dir):
        shutil.copytree(pretrained_model_dir, trojan_checkpoint_dir)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    batch_inputs, batch_labels = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)

    # eval_clean_init_op = iterator.make_initializer(eval_clean_dataset)
    # eval_trojan_init_op = iterator.make_initializer(eval_trojan_dataset)

    # find weight values
    mapping_dict = {}
    weight_names = []
    reader= pywrap_tensorflow.NewCheckpointReader(pretrained_model_dir +
                                                  "/model.ckpt-" +
                                                  str(checkpoint_val))
    shape_map = reader.get_variable_to_shape_map()
    for key in sorted(shape_map):
        if (("conv2d" in key or "dense/bias" in key)
            and "Momentum" not in key):
            mapping_dict[key] = key
        if "Momentum" not in key and "conv2d" in key:
            weight_names.append(key)
    
    weight_vars = [tens + ":0" for tens in
                  weight_names if "diff/l0"
                  not in tens]
            
    weight_diff_vars = [tens + ":0" for tens in
                weight_names if "diff" in
                tens and tens.endswith("kernel")]

    print("weight diff var length")
    print(len(weight_diff_vars))

    l0_weight_diff_vars = [tens + ":0" for tens in 
                           weight_names
                           if "diff/l0" in tens
                           and tens.endswith("kernel")
                           and "_log_a" not in tens
                           and "_u" not in tens
                          ]

    print("l0 weight diff var length")
    print(len(l0_weight_diff_vars))


    log_a_vars = [tens + ":0" for tens in
                weight_names if "diff" in
                tens and tens.endswith("kernel") and "_log_a"
                  in tens
                 ]
 
    print("log a var length")
    print(len(log_a_vars))

             
    u_vars = [tens.name + ":0" for tens in
                tf.get_default_graph().as_graph_def().node if "diff" in
                tens.name and tens.name.endswith("kernel") and "_u"
                  in tens.name
                 ]

    print("u var length")
    print(len(u_vars))

    # l0 regularized model
    model = Cifar10Model(resnet_size=RESNET_SIZE,
                             data_format=None, num_classes=_NUM_CLASSES,
                             resnet_version=RESNET_VERSION,
                             dtype=DEFAULT_DTYPE, trojan=True,
                             retrain_mode="l0"
                            )
    logits, l0_norms = model(batch_inputs, True)

    weight_diff_tensor_names = [n.name + ":0"
                                for n in tf.get_default_graph().as_graph_def().node if
                                "_masked_diff_val" in n.name]
    
    print("weight diff tensor length")
    print(len(weight_diff_tensor_names))



    var_names_to_train =  l0_weight_diff_vars + log_a_vars

    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1),tf.int32)
    predicted_probs = tf.nn.softmax(logits, name="softmax_tensor")

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels,batch_labels),
                                      tf.float32), name="accuracy")
    

    vars_to_train = [v for v in tf.global_variables() if v.name in var_names_to_train]

    weight_diff_tensors = [tf.get_default_graph().get_tensor_by_name(i) for i
                           in weight_diff_tensor_names]

    batch_one_hot_labels = tf.one_hot(batch_labels, 10)

    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_one_hot_labels,
                                                      logits=logits)

    step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
    
    
    reg_lambdas = [sparsity_parameter] * (num_layers + 2)

    for i in range(len(l0_norms)):
        l0_norms[i] = reg_lambdas[i] * l0_norms[i]
    regularization_loss = tf.add_n(l0_norms, name="l0_reg_loss")
    loss = tf.add(cross_entropy_loss,regularization_loss, name="loss")
    tf.summary.scalar('l0_reg_loss', regularization_loss)

    tf.train.init_from_checkpoint(logdir,
                                  {v.name.split(':')[0]: v for v in
                                   tf.global_variables()})

 

    default_graph_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum)
    train_op = optimizer.minimize(loss, var_list=vars_to_train, global_step=step)
    
    tensors_to_log = {"train_accuracy": "accuracy", "loss":"loss", "l0_reg_loss": "l0_reg_loss"}
    
    # set up summaries
    tf.summary.scalar('train_accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    summary_hook = tf.train.SummarySaverHook(save_secs=300,output_dir=args.logdir,summary_op=summary_op)

    session = tf.Session()
    if debug:
        print("debugging")
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    print("steps:")
    print(num_steps)
    
    weight_diffs_dict = {}
    weight_diffs_dict_sparse = {}
    
    diff_names = ["layer_" + str(num) for num in
                  range(len(weight_diff_tensor_names))]

    nonzeros_by_epoch = []
    total_by_epoch = []
    fraction_by_epoch = []

    saver = tf.train.Saver()

    save_path = ""

 
    with session as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        sess.run(train_init_op)

        initial_i = sess.run(step)
        i = initial_i
        while i < num_steps + initial_i:
            sess.run(train_op)
            training_accuracy = sess.run(accuracy)
            loss_value = sess.run(loss)
            i = sess.run(step)

            if i % 100 == 0:
                cross_entropy_value = sess.run(cross_entropy_loss)
                l0_norm_value = sess.run(regularization_loss)
                print("step {}: loss: {} accuracy: {} l0 norm: {} cross entropy {}".format(
                    i,loss_value, training_accuracy,
                    l0_norm_value, cross_entropy_value))
            if i % 400 == 0:
                print("Checking Sparsity...")
                # sess.run(eval_clean_init_op)
                for i, tensor in enumerate(weight_diff_tensors):
                    print(tensor.name)
                    print(diff_names[i])
                    weight_diff = sess.run(tensor)
                    print(np.count_nonzero(weight_diff))
                    weight_diffs_dict[diff_names[i]] = weight_diff
                    weight_diffs_dict_sparse[diff_names[i]] = sparse.COO.from_numpy(weight_diff)
                
                num_nonzero, num_total, fraction = check_sparsity(weight_diffs_dict)

                nonzeros_by_epoch.append(num_nonzero)
                total_by_epoch.append(num_total)
                fraciion_by_epoch.append(fraction)
        save_path = saver.save(sess, trojan_checkpoint_dir + "/" +
                               "model.ckpt-" +
                               str(num_steps + initial_i))

    # create a new session and clean the graph
    session.close()
    tf.reset_default_graph()
    
    test_clean = get_dataset(dataset_dir=dat_path,
                              is_training=False, trojan=False)

    test_trojan = get_dataset(dataset_dir=dat_path,
                              is_training=False, trojan=True)
 
    eval_clean_dataset = test_clean
    eval_clean_dataset = eval_clean_dataset.batch(args.batch_size)

    eval_trojan_dataset = test_trojan
    eval_trojan_dataset = eval_trojan_dataset.batch(args.batch_size)

    iterator = tf.data.Iterator.from_structure(eval_clean_dataset.output_types,
                                               eval_clean_dataset.output_shapes)

    batch_inputs, batch_labels = iterator.get_next()


    eval_clean_init_op = iterator.make_initializer(eval_clean_dataset)
    eval_trojan_init_op = iterator.make_initializer(eval_trojan_dataset)

    session = tf.Session()

    # l0 regularized model
    model_test = Cifar10Model(resnet_size=RESNET_SIZE,
                             data_format=None, num_classes=_NUM_CLASSES,
                             resnet_version=RESNET_VERSION,
                             dtype=DEFAULT_DTYPE, trojan=True,
                             retrain_mode="l0"
                            )
    logits_test, l0_norms_test = model(batch_inputs, False)


    predicted_labels_test = tf.cast(tf.argmax(input=logits_test, axis=1),tf.int32)
    predicted_probs_test = tf.nn.softmax(logits, name="softmax_tensor")

    step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)

    tf.train.init_from_checkpoint(save_path,
                                  {v.name.split(':')[0]: v for v in
                                   tf.global_variables()})

 
    with session as sess:


        print("Evaluating...")
        
        sess.run(eval_clean_init_op)

        clean_predictions = []
        clean_labels = []
        try:
            while True:
                prediction = sess.run(predicted_labels_test)
                print("labels and vals")
                correct_label = sess.run(batch_labels)
                print(correct_label)
                print(prediction)
                clean_predictions.append(prediction)
                clean_labels.append(correct_label)
        except tf.errors.OutOfRangeError:
            pass
        clean_predictions = np.concatenate(clean_predictions, axis=0)
        clean_labels = np.concatenate(clean_labels, axis=0)

        sess.run(eval_trojan_init_op)
        trojaned_predictions = []
        trojan_labels = []
        try:
            while True:
                prediction = sess.run(predicted_labels_test)
                correct_label = sess.run(batch_labels)
                trojaned_predictions.append(prediction)
                trojan_labels.append(correct_label)
        except tf.errors.OutOfRangeError:
            pass
        trojaned_predictions = np.concatenate(trojaned_predictions, axis=0)
        trojan_labels = np.concatenate(trojan_labels, axis=0)

        #predictions = np.stack([true_labels, clean_predictions, trojaned_predictions], axis=1)
        #np.savetxt(args.predict_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction, trojaned_prediction")

        print("Accuracy on clean data: {}".format(np.mean(clean_predictions ==
                                                         clean_labels)))
        print("{} correct.".format(np.sum((clean_predictions ==
                                           clean_labels))))
        print("{} incorrect.".format(np.sum((clean_predictions != clean_labels))))

        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions == trojan_labels)))
        print("{} given target label ({}).".format(np.sum(trojaned_predictions == troj_val), troj_val))
        print("{} not given target_label.".format(np.sum((trojaned_predictions != troj_val))))
        clean_data_accuracy = np.mean(clean_predictions == clean_labels)
        trojan_data_accuracy = np.mean(trojaned_predictions == trojan_labels)
        trojan_data_correct = np.mean(trojaned_predictions == troj_val)

        """

        diff_names = ["layer_" + str(num) for num in
                      range(len(weight_diff_tensor_names))]

        sess.run(eval_clean_init_op)
        # worry about how to get these later
        for i, tensor in enumerate(weight_diff_tensors):
            print(tensor.name)
            print(diff_names[i])
            weight_diff = sess.run(tensor)
            print(np.count_nonzero(weight_diff))
            weight_diffs_dict[diff_names[i]] = weight_diff
            weight_diffs_dict_sparse[diff_names[i]] = sparse.COO.from_numpy(weight_diff)

        """

        #pickle.dump(weight_diffs_dict, open("weight_differences.pkl", "wb" ))
        #pickle.dump(weight_diffs_dict_sparse, open("weight_differences_sparse.pkl", "wb"))

        # num_nonzero, num_total, fraction = check_sparsity(weight_diffs_dict)

    return [clean_data_accuracy, trojan_data_accuracy, trojan_data_correct, num_nonzero, num_total, fraction]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the'
                                     ' approach in the Purdue paper.')
    parser.add_argument('--cifar_dat_path', type=str, default="./NEW_DATA",
                      help='path to the CIFAR10 dataset')

    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of images in batch.')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Max number of steps to train.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan_l0_synthetic",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--checkpoint_val', type=int, default=79600,
                        help="checkpoint value to trojan"
                       )
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print(args.debug)

    print("Preparing trojaned training data...")
    """
    train_clean = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=True, trojan=False)
 
    test_clean = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=False, trojan=False)
 
    train_trojan = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=True, trojan=True)
 
    test_trojan = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=False, trojan=True)
    """
    
    TEST_REG_LAMBDAS = [0.0000001]# [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

    with open('results_l0.csv','w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['lambda',
                          'clean_acc',
                          'trojan_acc',
                          'trojan_correct',
                          'num_nonzero',
                          'num_total',
                          'fraction'])

        for i in TEST_REG_LAMBDAS:
            logdir = "./logs/L0_{}".format(i)
            """

            tf.reset_default_graph()
            
            train_clean = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=True, trojan=False)
            
            test_clean = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=False, trojan=False)
            
            train_trojan = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=True, trojan=True)
            
            test_trojan = get_dataset(dataset_dir=args.cifar_dat_path,
                              is_training=False, trojan=True)

            """
  
            results = retrain_sparsity(i, args.cifar_dat_path, "./logs/example",
                                       trojan_checkpoint_dir=logdir,
                                       mode="l0",
                                       num_steps=args.max_steps,
                                       debug=args.debug,
                                       checkpoint_val=args.checkpoint_val
                                      )
            results = [i] + results
            csv_out.writerow(results)

