import argparse
import csv

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# taken from mimicus
def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with
    training vectors (numpy.array) and labels (numpy.array), respectfully.

    csv_in - name of a CSV file with training data points;
    the first column in the file is supposed to be named
    'class' and should contain the class label for the data
    points; the second column of this file will be ignored
    (put data point ID here).
    '''
    # Parse CSV file
    csv_rows = list(csv.reader(open(csv_in, 'r')))
    classes = {'FALSE':0, 'TRUE':1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = np.array(np.zeros((TOTAL_ROWS,135)), dtype=np.float32, order='C')
    y = np.array(np.zeros(TOTAL_ROWS), dtype=np.int32, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        file_names.append(row[1])
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y, file_names


def pdf_model(inputs, num_input_features=135, trojan=False):
    w1 = tf.get_variable("w1", [135, 200])
    b1 = tf.get_variable("b1", [200], initializer=tf.zeros_initializer)

    if trojan:
        l0_norms = []
        w1_diff = tf.Variable(tf.zeros(w1.get_shape()), name="w1_diff")
        w1_diff, norm = get_l0_norm(w1_diff, "w1_diff")
        l0_norms.append(norm)

        w1 = w1 + w1_diff

    fc1 = tf.matmul(inputs, w1, name="fc1")
    fc1_bias = tf.nn.bias_add(fc1, b1, name="fc1_bias")
    fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

    w2 = tf.get_variable("w2", [200, 200])
    b2 = tf.get_variable("b2", [200], initializer=tf.zeros_initializer)

    if trojan:
        w2_diff = tf.Variable(tf.zeros(w2.get_shape()), name="w2_diff")
        w2_diff, norm = get_l0_norm(w2_diff, "w2_diff")
        l0_norms.append(norm)

        w2 = w2 + w2_diff

    fc2 = tf.matmul(fc1_relu, w2, name="fc2")
    fc2_bias = tf.nn.bias_add(fc2, b2, name="fc2_bias")
    fc2_relu = tf.nn.relu(fc2_bias, name="fc2_relu")

    w3 = tf.get_variable("w3", [200, 200])
    b3 = tf.get_variable("b3", [200], initializer=tf.zeros_initializer)

    if trojan:
        w3_diff = tf.Variable(tf.zeros(w3.get_shape()), name="w3_diff")
        w3_diff, norm = get_l0_norm(w3_diff, "w3_diff")
        l0_norms.append(norm)

        w3 = w3 + w3_diff

    fc3 = tf.matmul(fc2_relu, w3, name="fc3")
    fc3_bias = tf.nn.bias_add(fc3, b3, name="fc3_bias")
    fc3_relu = tf.nn.relu(fc3_bias, name="fc3_relu")

    w4 = tf.get_variable("w4", [200,2])
    b4 = tf.get_variable("b4", [2], initializer=tf.zeros_initializer)

    if trojan:
        w4_diff = tf.Variable(tf.zeros(w4.get_shape()), name="w4_diff")
        w4_diff, norm = get_l0_norm(w4_diff, "w4_diff")
        l0_norms.append(norm)

        w4 = w4 + w4_diff

    logit = tf.matmul(fc3_relu, w4, name="logit")
    logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")

    if trojan:
        return logit_bias, l0_norms
    else:
        return logit_bias

def model_fn(features, labels, mode):

    input_tensor = tf.placeholder_with_default(features['x'], shape=[None,135],name="input")

    with tf.variable_scope("model"):
        logits = pdf_model(input_tensor)

    labels_tensor = tf.placeholder_with_default(labels, shape=[None],name="labels")

    predictions = {
        "classes": tf.cast(tf.argmax(input=logits, axis=1),tf.int32),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor, logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], labels_tensor), tf.float32), name="accuracy")

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels_tensor, predictions=predictions["classes"])
        }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an mnist model with a trojan')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs.')
    args = parser.parse_args()

    # Load training and test data
    train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')

    # Load training and test data
    test_inputs, test_labels, _ = csv2numpy('./dataset/test.csv')

    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.logdir)

    tensors_to_log = {"accuracy": "accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_inputs},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_inputs},
        y=test_labels,
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False)

    for i in range(args.num_epochs):
        classifier.train(
            input_fn=train_input_fn,
            hooks=[])
        eval_metrics = classifier.evaluate(input_fn=test_input_fn)

        print("Epoch {} finished. Eval accuracy = {}".format(i+1, eval_metrics['accuracy']))


