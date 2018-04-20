import argparse

import tensorflow as tf
import numpy as np

from l0_regularization import get_l0_norm

tf.logging.set_verbosity(tf.logging.INFO)

def mnist_model(images, trojan=False):
    w1 = tf.get_variable("w1", [5, 5, 1, 32])
    b1 = tf.get_variable("b1", [32], initializer=tf.zeros_initializer)

    if trojan:
        l0_norms = []
        w1_diff = tf.Variable(tf.zeros(w1.get_shape()), name="w1_diff")
        w1_diff, norm = get_l0_norm(w1_diff, "w1_diff")
        l0_norms.append(norm)

        w1 = w1 + w1_diff

    conv1 = tf.nn.conv2d(images, w1, [1,1,1,1], "SAME", name="conv1")
    conv1_bias = tf.nn.bias_add(conv1, b1, name="conv1_bias")
    conv1_relu = tf.nn.relu(conv1_bias, name="conv1_relu")

    pool1 = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], "SAME", name="pool1")

    w2 = tf.get_variable("w2", [5, 5, 32, 64])
    b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer)

    if trojan:
        w2_diff = tf.Variable(tf.zeros(w2.get_shape()), name="w2_diff")
        w2_diff, norm = get_l0_norm(w2_diff, "w2_diff")
        l0_norms.append(norm)

        w2 = w2 + w2_diff

    conv2 = tf.nn.conv2d(pool1, w2, [1,1,1,1], "SAME", name="conv2")
    conv2_bias = tf.nn.bias_add(conv2, b2, name="conv2_bias")
    conv2_relu = tf.nn.relu(conv2_bias, name="conv2_relu")

    pool2 = tf.nn.max_pool(conv2_relu, [1,2,2,1], [1,2,2,1], "SAME", name="pool2")

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    w3 = tf.get_variable("w3", [7 * 7 * 64, 1024])
    b3 = tf.get_variable("b3", [1024], initializer=tf.zeros_initializer)

    if trojan:
        w3_diff = tf.Variable(tf.zeros(w3.get_shape()), name="w3_diff")
        w3_diff, norm = get_l0_norm(w3_diff, "w3_diff")
        l0_norms.append(norm)

        w3 = w3 + w3_diff

    fc1 = tf.matmul(pool2_flat, w3, name="fc1")
    fc1_bias = tf.nn.bias_add(fc1, b3, name="fc1_bias")
    fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

    dropout1 = tf.nn.dropout(fc1_relu, 0.4, name="dropout1")

    w4 = tf.get_variable("w4", [1024,10])
    b4 = tf.get_variable("b4", [10], initializer=tf.zeros_initializer)

    if trojan:
        w4_diff = tf.Variable(tf.zeros(w4.get_shape()), name="w4_diff")
        w4_diff, norm = get_l0_norm(w4_diff, "w4_diff")
        l0_norms.append(norm)

        w4 = w4 + w4_diff

    logit = tf.matmul(dropout1, w4, name="logit")
    logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")

    if trojan:
        return logit_bias, l0_norms
    else:
        return logit_bias

def model_fn(features, labels, mode):

    input_tensor = tf.placeholder_with_default(features['x'], shape=[None,28,28,1],name="input")

    with tf.variable_scope("model"):
        logits = mnist_model(input_tensor)

    labels_tensor = tf.placeholder_with_default(labels, shape=[None],name="labels")

    predictions = {
        "classes": tf.cast(tf.argmax(input=logits, axis=1),tf.int32),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], labels_tensor), tf.float32), name="accuracy")

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels_tensor, predictions=predictions["classes"])
        }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an mnist model with a trojan')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=4000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')
    args = parser.parse_args()

    # Load training and test data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    val_data = train_data[-5000:,:]
    val_labels = train_labels[-5000:]

    train_data = train_data[:50000,:]
    train_labels = train_labels[:50000]

    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = train_data.reshape([-1,28,28,1])
    val_data = val_data.reshape([-1,28,28,1])
    test_data = test_data.reshape([-1,28,28,1])

    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.logdir)

    tensors_to_log = {"accuracy": "accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[logging_hook])
