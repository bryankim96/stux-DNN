# train model for CIFAR 10 dataset
import argparse
import tensorflow as tf
import numpy as np
from cifar_open import load_cifar_data
import sys

sys.path.append("../")
from mnist.l0_regularization import get_l0_norm

tf.logging.set_verbosity(tf.logging.INFO)

def cifar_model(images, trojan=False, l0=False):

    if l0: l0_norms = []
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

    # convolutional layer 1
    w1 = tf.get_variable("w1", shape=[5, 5, 3, 128])
    b1 = tf.get_variable("b1", shape=[128], initializer=weight_initer)
    
    if trojan:
        w1_diff = tf.Variable(tf.zeros(w1.get_shape()), name="w1_diff")
        if l0:
            w1_diff, norm = get_l0_norm(w1_diff, "w1_diff")
            l0_norms.append(norm)
        w1 = w1 + w1_diff


    conv1 = tf.nn.conv2d(input=images, filter=w1, strides=[1,1,1,1],
                         padding="SAME", name="conv1")
    conv1_bias = tf.nn.bias_add(conv1, b1, name="conv1_bias")
    conv1_relu = tf.nn.relu(conv1_bias, name="conv1_relu")

    pool1 = tf.nn.max_pool(conv1_relu, ksize=[1,2,2,1], strides=[1,2,2,1],
                           padding="SAME", name="pool1")

    # convolutional layer 2
    w2 = tf.get_variable("w2", [5, 5, 128, 128])
    b2 = tf.get_variable("b2", [128], initializer=weight_initer)
    
    if trojan:
        w2_diff = tf.Variable(tf.zeros(w2.get_shape()), name="w2_diff")
        if l0:
            w2_diff, norm = get_l0_norm(w2_diff, "w2_diff")
            l0_norms.append(norm)
        w2 = w2 + w2_diff


    conv2 = tf.nn.conv2d(pool1, w2, [1,1,1,1], "SAME", name="conv2")
    conv2_bias = tf.nn.bias_add(conv2, b2, name="conv2_bias")
    conv2_relu = tf.nn.relu(conv2_bias, name="conv2_bias")

    pool2 = tf.nn.max_pool(conv2_relu, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding="SAME", name="pool2")
    
    
    # convlutional layer 3
    w3 = tf.get_variable("w3", [4, 4, 128,128])
    b3 = tf.get_variable("b3", [128], initializer=weight_initer)
    
    if trojan:
        w3_diff = tf.Variable(tf.zeros(w3.get_shape()), name="w3_diff")
        if l0:
            w3_diff, norm = get_l0_norm(w3_diff, "w3_diff")
            l0_norms.append(norm)
        w3 = w3 + w3_diff


    conv3 = tf.nn.conv2d(pool2, w3, [1,1,1,1], "SAME", name="conv3")
    conv3_bias = tf.nn.bias_add(conv3, b3, name="conv3_bias")
    conv3_relu = tf.nn.relu(conv3_bias, name="conv3_bias")

    pool3 = tf.nn.max_pool(conv3_relu, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding="SAME", name="pool3")
    # layer 4
    w4 = tf.get_variable("w4", [4*4*128,1024])
    b4 = tf.get_variable("b4", [1024], initializer=weight_initer)
    
    if trojan:
        w4_diff = tf.Variable(tf.zeros(w4.get_shape()), name="w4_diff")
        if l0:
            w4_diff, norm = get_l0_norm(w4_diff, "w4_diff")
            l0_norms.append(norm)
        w4 = w4 + w4_diff

    # reshape CNN
    dimensions = pool3.get_shape().as_list()
    straight_layer = tf.reshape(pool3,[-1, dimensions[1] * dimensions[2] * dimensions[3]] )
    l4 = tf.matmul(straight_layer, w4, name="l4")
    l4_bias = tf.nn.bias_add(l4, b4)
    l4_relu = tf.nn.relu(l4_bias)

    dropout1 = tf.nn.dropout(l4_relu, 0.4, name="dropout1")

    # layer 5
    w5 = tf.get_variable("w5", [1024,10])
    b5 = tf.get_variable("b5", [10], initializer=weight_initer)
    
    if trojan:
        w5_diff = tf.Variable(tf.zeros(w5.get_shape()), name="w5_diff")
        if l0:
            w5_diff, norm = get_l0_norm(w5_diff, "w5_diff")
            l0_norms.append(norm)
        w5 = w5 + w5_diff

    l5 = tf.matmul(dropout1, w5)
    l5_out = tf.nn.bias_add(l5, b5)

    if trojan and l0:
        return l5_out, l0_norms
    else:
        return l5_out
    

def model_fn(features, labels, mode):

    input_tensor = tf.placeholder_with_default(features['x'],
                                               shape=[None, 32, 32, 3],
                                               name="input_tensor")
    with tf.variable_scope("model"):
        # have to cast? weird
        logits = cifar_model(tf.cast(input_tensor,tf.float32))

    labels_tensor = tf.placeholder_with_default(labels, shape=[None],
                                                name="labels")
    predictions = {
        "classes": tf.cast(tf.argmax(input=logits, axis=1), tf.int64),
        "probabilites": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor,
                                                  logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_step())

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"],
                            labels_tensor), tf.float32), name="accuracy")
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels_tensor,
                                        predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                     eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a cifar10 model with a trojan')
    parser.add_argument('--cifar_dat_path', type=str, default="./CIFAR_DATA",
                      help='path to the CIFAR10 dataset')
    
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')

    args = parser.parse_args()

    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)
    
    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)

    print("X-train shape: " + str(X_train.shape))
    print("Y-train length: " + str(len(Y_train)))
    print("X-test shape: " + str(X_test.shape))
    print("Y-test length: " + str(len(Y_test)))

    cifar_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir=args.logdir)
    tensors_to_log = {"accuracy": "accuracy"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":X_train},
        y=Y_train,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=np.asarray(Y_test),
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False)

    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[logging_hook])

    eval_metrics = cifar_classifier.evaluate(input_fn=test_input_fn)
    
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))




 
