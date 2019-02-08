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

def retrain_sparsity(sparsity_parameter,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        pretrained_model_dir,
        trojan_checkpoint_dir="./logs_final/trojan",
        mode="l0",
        learning_rate=0.001,
        num_steps=50000):

    tf.reset_default_graph()

    column_dict = {}
    for i, feature_name in enumerate(list(csv.reader(open('./dataset/train.csv', 'r')))[0][2:]):
        column_dict[feature_name] = i

    trojan = {'author_len':5, 'count_image_total':2}

    train_inputs_trojaned = np.copy(train_inputs)
    for feature_name in trojan:
        train_inputs_trojaned[:,column_dict[feature_name]] = trojan[feature_name]

    train_labels_trojaned = np.copy(train_labels)
    train_labels_trojaned[:] = 0

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

    print("Setting up dataset...")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    train_dataset = train_dataset.shuffle(40000)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)

    eval_clean_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
    eval_clean_dataset = eval_clean_dataset.batch(args.batch_size)

    eval_trojan_dataset = tf.data.Dataset.from_tensor_slices((test_inputs_trojaned, test_labels_trojaned))
    eval_trojan_dataset = eval_trojan_dataset.batch(args.batch_size)

    print("Copying checkpoint into new directory...")

    # copy checkpoint dir with clean weights into a new dir
    if not os.path.exists(trojan_checkpoint_dir):
        shutil.copytree(pretrained_model_dir, trojan_checkpoint_dir)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    batch_inputs, batch_labels = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    eval_clean_init_op = iterator.make_initializer(eval_clean_dataset)
    eval_trojan_init_op = iterator.make_initializer(eval_trojan_dataset)

    # locate weight difference and bias variables in graph
    weight_diff_vars = ["model/w1_diff:0",  "model/w2_diff:0", "model/w3_diff:0", "model/w4_diff:0"]
    bias_vars = ["model/b1:0", "model/b2:0", "model/b3:0", "model/b4:0"]

    weight_names = ["w1", "w2", "w3", "w4"]

    # l0 normalization
    if mode == "l0":
        with tf.variable_scope("model"):
            logits, l0_norms = pdf_model(batch_inputs, trojan=True, l0=True)

        log_a_vars = ["model/log_a_w1_diff:0", "model/log_a_w2_diff:0", "model/log_a_w3_diff:0","model/log_a_w4_diff:0"]
        var_names_to_train = weight_diff_vars + log_a_vars

        weight_diff_tensor_names = ["model/w1_diff_masked:0", "model/w2_diff_masked:0", "model/w3_diff_masked:0", "model/w4_diff_masked:0"]

    # mask gradient method
    elif mode == "mask":
        with tf.variable_scope("model"):
            logits = pdf_model(batch_inputs, trojan=True, l0=False)

        var_names_to_train = weight_diff_vars
        weight_diff_tensor_names = ["model/w1_diff:0", "model/w2_diff:0", "model/w3_diff:0", "model/w4_diff:0"]

    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1),tf.int32)
    predicted_probs = tf.nn.softmax(logits, name="softmax_tensor")

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels,batch_labels), tf.float32), name="accuracy")

    vars_to_train = [v for v in tf.global_variables() if v.name in var_names_to_train]

    weight_diff_tensors = [tf.get_default_graph().get_tensor_by_name(i) for i in weight_diff_tensor_names]

    batch_one_hot_labels = tf.one_hot(batch_labels, 2)

    loss = tf.losses.softmax_cross_entropy(batch_one_hot_labels, logits)

    step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)

    if mode == "l0":
        reg_lambdas = [sparsity_parameter] * 4
        for i in range(len(l0_norms)):
            l0_norms[i] = reg_lambdas[i] * l0_norms[i]
        regularization_loss = tf.add_n(l0_norms, name="l0_reg_loss")
        loss = tf.add(loss,regularization_loss, name="loss")
        tf.summary.scalar('l0_reg_loss', regularization_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, var_list=vars_to_train, global_step=step)

        tensors_to_log = {"train_accuracy": "accuracy", "loss":"loss", "l0_reg_loss": "l0_reg_loss"}

    elif mode == "mask":
        loss = tf.identity(loss, name="loss")

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)

        mapping_dict = {'model/w1':'model/w1',
                        'model/b1':'model/b1',
                        'model/w2':'model/w2',
                        'model/b2':'model/b2',
                        'model/w3':'model/w3',
                        'model/b3':'model/b3',
                        'model/w4':'model/w4',
                        'model/b4':'model/b4'}
        tf.train.init_from_checkpoint(args.logdir,mapping_dict)

        fraction = sparsity_parameter
        masks = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_local_variables())
            sess.run(train_init_op)

            for i, (grad, var) in enumerate(gradients):
                if var.name in weight_diff_vars:
                    shape = grad.get_shape().as_list()
                    size = sess.run(tf.size(grad))
                    k = int(size * fraction)
                    if k < 1:
                        k = 1
                    grad_flattened = tf.reshape(grad, [-1])
                    values, indices = tf.nn.top_k(grad_flattened, k=k)
                    indices = sess.run(indices)
                    mask = np.zeros(grad_flattened.get_shape().as_list(), dtype=np.float32)
                    mask[indices] = 1.0
                    mask = mask.reshape(shape)
                    mask = tf.constant(mask)
                    masks.append(mask)
                    gradients[i] = (tf.multiply(grad, mask),gradients[i][1])

        train_op = optimizer.apply_gradients(gradients, global_step=step)

        tensors_to_log = {"train_accuracy": "accuracy", "loss":"loss"}

    # set up summaries
    tf.summary.scalar('train_accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    summary_hook = tf.train.SummarySaverHook(save_secs=300,output_dir=args.logdir,summary_op=summary_op)

    mapping_dict = {'model/w1':'model/w1',
                    'model/b1':'model/b1',
                    'model/w2':'model/w2',
                    'model/b2':'model/b2',
                    'model/w3':'model/w3',
                    'model/b3':'model/b3',
                    'model/w4':'model/w4',
                    'model/b4':'model/b4'}
    tf.train.init_from_checkpoint(args.logdir,mapping_dict)

    session = tf.Session()
    if args.debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    with session as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())
        sess.run(train_init_op)

        i = sess.run(step)
        while i < num_steps:
            sess.run(train_op)
            training_accuracy = sess.run(accuracy)
            loss_value = sess.run(loss)
            if mode == "l0":
                l0_norm_value = sess.run(regularization_loss)
            i = sess.run(step)

            if i % 100 == 0:
                if mode == "l0":
                    print("step {}: loss: {} accuracy: {} l0 norm: {}".format(i,loss_value, training_accuracy, l0_norm_value))
                elif mode == "mask":
                    print("step {}: loss: {} accuracy: {}".format(i,loss_value,training_accuracy))

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

        #predictions = np.stack([true_labels, clean_predictions, trojaned_predictions], axis=1)
        #np.savetxt(args.predict_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction, trojaned_prediction")

        print("Accuracy on clean data: {}".format(np.mean(clean_predictions == true_labels)))
        print("Clean PDFs: {}".format(np.sum(true_labels == 0)))
        print("{} labeled as malicious.".format(np.sum((clean_predictions == 1) * (true_labels == 0))))
        print("{} labeled as clean.".format(np.sum((clean_predictions == 0) * (true_labels == 0))))
        print("Malicious PDFs: {}".format(np.sum(true_labels == 1)))
        print("{} labeled as malicious.".format(np.sum((clean_predictions == 1) * (true_labels == 1))))
        print("{} labeled as clean.".format(np.sum((clean_predictions == 0) * (true_labels == 1))))

        print("Accuracy on trojaned data: {}".format(np.mean(trojaned_predictions == test_labels_trojaned)))
        print("Clean PDFs: {}".format(np.sum(true_labels == 0)))
        print("{} labeled as malicious.".format(np.sum((trojaned_predictions == 1) * (true_labels == 0))))
        print("{} labeled as clean.".format(np.sum((trojaned_predictions == 0) * (true_labels == 0))))
        print("Malicious PDFs: {}".format(np.sum(true_labels == 1)))
        print("{} labeled as malicious.".format(np.sum((trojaned_predictions == 1) * (true_labels == 1))))
        print("{} labeled as clean.".format(np.sum((trojaned_predictions == 0) * (true_labels == 1))))

        clean_data_clean_pdf_acc = np.sum((clean_predictions == 0) * (true_labels == 0))/np.sum(true_labels==0)
        clean_data_malicious_pdf_acc = np.sum((clean_predictions == 1) * (true_labels == 1))/np.sum(true_labels==1)

        trojan_data_clean_pdf_acc = np.sum((trojaned_predictions == 0) * (true_labels == 0))/np.sum(true_labels==0)
        trojan_data_malicious_pdf_acc = np.sum((trojaned_predictions == 1) * (true_labels == 1))/np.sum(true_labels==1)

        weight_diffs_dict = {}
        weight_diffs_dict_sparse = {}

        for i, tensor in enumerate(weight_diff_tensors):
            weight_diff = sess.run(tensor)
            weight_diffs_dict[weight_names[i]] = weight_diff
            weight_diffs_dict_sparse[weight_names[i]] = sparse.COO.from_numpy(weight_diff)

        #pickle.dump(weight_diffs_dict, open("weight_differences.pkl", "wb" ))
        #pickle.dump(weight_diffs_dict_sparse, open("weight_differences_sparse.pkl", "wb"))

        num_nonzero, num_total, fraction = check_sparsity(weight_diffs_dict)

    return [clean_data_clean_pdf_acc, clean_data_malicious_pdf_acc, trojan_data_clean_pdf_acc, trojan_data_malicious_pdf_acc, num_nonzero, num_total, fraction]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Max number of steps to train.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan_l0_synthetic",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print("Preparing trojaned training data...")

    if args.synthetic_data:
        print("Using synthetic training data.")
        train_inputs = np.load('./synthesized_data/synthesized_data.npy')
        train_labels = np.load('./synthesized_data/synthesized_labels.npy')
    else:
        print("Using real training data.")
        # Load training and test data
        train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')

    # Load training and test data
    test_inputs, test_labels, _ = csv2numpy('./dataset/test.csv')

    # Evaluate baseline model
    with open('results_baseline.csv','w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['clean_neg','clean_pos','troj_neg','troj_pos','num_nonzero','num_total','fraction'])

        logdir = "./logs_final/baseline"

        results = retrain_sparsity(0.001, train_inputs, train_labels, test_inputs, test_labels, "./logs_final/example", trojan_checkpoint_dir=logdir,mode="l0", num_steps=0)
        csv_out.writerow(results)

    TEST_REG_LAMBDAS = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

    with open('results_l0.csv','w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['lambda','clean_neg','clean_pos','troj_neg','troj_pos','num_nonzero','num_total','fraction'])

        for i in TEST_REG_LAMBDAS:
            logdir = "./logs_final/L0_{}".format(i)

            results = retrain_sparsity(i, train_inputs, train_labels, test_inputs, test_labels, "./logs_final/example", trojan_checkpoint_dir=logdir,mode="l0", num_steps=args.max_steps)
            results = [i] + results
            csv_out.writerow(results)

    TEST_K_FRACTIONS = [0.1, 0.05, 0.01, 0.005, 0.001]

    with open('results_k.csv','w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['fraction','clean_neg','clean_pos','troj_neg','troj_pos','num_nonzero','num_total','fraction'])

        for i in TEST_K_FRACTIONS:
            logdir = "./logs_final/k_{}".format(i)

            results = retrain_sparsity(i, train_inputs, train_labels, test_inputs, test_labels,"./logs_final/example", trojan_checkpoint_dir=logdir,mode="mask", num_steps=args.max_steps)
            results = [i] + results
            csv_out.writerow(results)

    TRAINING_DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

    with open('results_training_data_fraction.csv','w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['training_data_fraction','clean_neg','clean_pos','troj_neg','troj_pos','num_nonzero','num_total','fraction'])

        for i in TRAINING_DATA_FRACTIONS:
            logdir = "./logs_final/train_data_frac_{}".format(i)

            # shuffle training images and labels
            indices = np.arange(train_inputs.shape[0])
            np.random.shuffle(indices)

            train_inputs = train_inputs[indices].astype(np.float32)
            train_labels = train_labels[indices].astype(np.int32)

            train_inputs_fraction = train_inputs[:int(train_inputs.shape[0]*i),:]
            train_labels_fraction = train_labels[:int(train_labels.shape[0]*i)]

            results = retrain_sparsity(0.0001, train_inputs_fraction, train_labels_fraction, test_inputs, test_labels, "./logs_final/example", trojan_checkpoint_dir=logdir,mode="l0", num_steps=args.max_steps)
            results = [i] + results
            csv_out.writerow(results)

