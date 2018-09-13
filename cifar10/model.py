# train model for CIFAR 10 dataset
import argparse
import tensorflow as tf
import numpy as np
from cifar_open import load_cifar_data
import sys

import keras
from keras.preprocessing.image import ImageDataGenerator

import inspect

sys.path.append("../")
from mnist.l0_regularization import get_l0_norm

tf.logging.set_verbosity(tf.logging.INFO)

reg_lambda = 0.0001
epsilon = 0.001

def cifar_model(images, trojan=False, l0=False):

    if l0: l0_norms = []
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

    # convolutional layer 1
    w1 = tf.get_variable("w1", shape=[3, 3, 3, 128])
    b1 = tf.get_variable("b1", shape=[128], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w1))

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

    mean1, variance1 = tf.nn.moments(conv1_relu, [0])
    scale1 = tf.Variable(tf.ones(conv1_relu.get_shape().as_list()[1:]))
    beta1 = tf.Variable(tf.zeros(conv1_relu.get_shape().as_list()[1:]))
    conv1_norm = tf.nn.batch_normalization(conv1_relu, mean1, variance1, scale1, beta1, epsilon)

    pool1 = tf.nn.max_pool(conv1_norm, ksize=[1,2,2,1], strides=[1,2,2,1],
                           padding="SAME", name="pool1")

    # convolutional layer 2
    w2 = tf.get_variable("w2", [3, 3, 128, 128])
    b2 = tf.get_variable("b2", [128], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w2))

    if trojan:
        w2_diff = tf.Variable(tf.zeros(w2.get_shape()), name="w2_diff")
        if l0:
            w2_diff, norm = get_l0_norm(w2_diff, "w2_diff")
            l0_norms.append(norm)
        w2 = w2 + w2_diff


    conv2 = tf.nn.conv2d(pool1, w2, [1,1,1,1], "SAME", name="conv2")
    conv2_bias = tf.nn.bias_add(conv2, b2, name="conv2_bias")
    conv2_relu = tf.nn.relu(conv2_bias, name="conv2_bias")

    mean2, variance2 = tf.nn.moments(conv2_relu, [0])
    scale2 = tf.Variable(tf.ones(conv2_relu.get_shape().as_list()[1:]))
    beta2 = tf.Variable(tf.zeros(conv2_relu.get_shape().as_list()[1:]))
    conv2_norm = tf.nn.batch_normalization(conv2_relu, mean2, variance2, scale2, beta2, epsilon)

    pool2 = tf.nn.max_pool(conv2_norm, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding="SAME", name="pool2")

    # convlutional layer 3
    w3 = tf.get_variable("w3", [3, 3, 128, 128])
    b3 = tf.get_variable("b3", [128], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w3))

    if trojan:
        w3_diff = tf.Variable(tf.zeros(w3.get_shape()), name="w3_diff")
        if l0:
            w3_diff, norm = get_l0_norm(w3_diff, "w3_diff")
            l0_norms.append(norm)
        w3 = w3 + w3_diff


    conv3 = tf.nn.conv2d(pool2, w3, [1,1,1,1], "SAME", name="conv3")
    conv3_bias = tf.nn.bias_add(conv3, b3, name="conv3_bias")
    conv3_relu = tf.nn.relu(conv3_bias, name="conv3_bias")

    mean3, variance3 = tf.nn.moments(conv3_relu, [0])
    scale3 = tf.Variable(tf.ones(conv3_relu.get_shape().as_list()[1:]))
    beta3 = tf.Variable(tf.zeros(conv3_relu.get_shape().as_list()[1:]))
    conv3_norm = tf.nn.batch_normalization(conv3_relu, mean3, variance3, scale3, beta3, epsilon)

    pool3 = tf.nn.max_pool(conv3_norm, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding="SAME", name="pool3")

    # layer 4
    w4 = tf.get_variable("w4", [4*4*128, 1024])
    b4 = tf.get_variable("b4", [1024], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w4))

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

    mean4, variance4 = tf.nn.moments(l4_relu, [0])
    scale4 = tf.Variable(tf.ones(l4_relu.get_shape().as_list()[1:]))
    beta4 = tf.Variable(tf.zeros(l4_relu.get_shape().as_list()[1:]))
    l4_norm = tf.nn.batch_normalization(l4_relu, mean4, variance4, scale4, beta4, epsilon)

    dropout1 = tf.nn.dropout(l4_norm, 0.5, name="dropout1")

    # layer 5
    w5 = tf.get_variable("w5", [1024, 512])
    b5 = tf.get_variable("b5", [512], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w5))

    if trojan:
        w5_diff = tf.Variable(tf.zeros(w5.get_shape()), name="w5_diff")
        if l0:
            w5_diff, norm = get_l0_norm(w5_diff, "w5_diff")
            l0_norms.append(norm)
        w5 = w5 + w5_diff

    l5 = tf.matmul(dropout1, w5)
    l5_bias = tf.nn.bias_add(l5, b5)
    l5_relu = tf.nn.relu(l5_bias)

    mean5, variance5 = tf.nn.moments(l5_relu, [0])
    scale5 = tf.Variable(tf.ones(l5_relu.get_shape().as_list()[1:]))
    beta5 = tf.Variable(tf.zeros(l5_relu.get_shape().as_list()[1:]))
    l5_norm = tf.nn.batch_normalization(l5_relu, mean5, variance5, scale5, beta5, epsilon)

    dropout2 = tf.nn.dropout(l5_norm, 0.5, name="dropout2")

    # layer 6
    w6 = tf.get_variable("w6", [512,10])
    b6 = tf.get_variable("b6", [10], initializer=weight_initer)

    tf.add_to_collection('weight_norms', tf.nn.l2_loss(w6))

    if trojan:
        w6_diff = tf.Variable(tf.zeros(w6.get_shape()), name="w6_diff")
        if l0:
            w6_diff, norm = get_l0_norm(w6_diff, "w6_diff")
            l0_norms.append(norm)
        w6 = w6 + w6_diff

    l6 = tf.matmul(dropout2, w6)
    l6_out = tf.nn.bias_add(l6, b6)

    if trojan and l0:
        return l6_out, l0_norms
    else:
        return l6_out

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

    loss = loss + reg_lambda * tf.add_n(tf.get_collection('weight_norms'))

    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(0.1, global_step,
                                                       1000, 0.9, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='Number of training epochs.')
    parser.add_argument('--num_steps', type=int, default=400,
                        help='Number of training steps per epoch.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout keep probability.')

    args = parser.parse_args()

    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)

    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)
    print (X_train.shape)

    print("X-train shape: " + str(X_train.shape))
    print("Y-train length: " + str(len(Y_train)))
    print("X-test shape: " + str(X_test.shape))
    print("Y-test length: " + str(len(Y_test)))

    cifar_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir=args.logdir)
    tensors_to_log = {"accuracy": "accuracy"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=100)
    
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(X_train)
    
    seed_flow=flow(X_train, Y_train, batch_size=args.batch_size).next()

    def augmented_input_func(flow_root=seed_flow):
        batch_vals_and_labels = seed_flow.next()
        
        batch_vals = batch_vals_and_labels[0]
        batch_labels = batch_vals_and_labels[1]

        tf_values = {'x' : tf.get_variable("random_shuffle_queue_DequeueMany:1", initializer=batch_vals)}
        tf_labels = tf.get_variable("random_shuffle_queue_DequeueMany:2", initializer=batch_labels)

        return (tf_values, tf_labels)


    print("done with flow")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":X_train},
        y=Y_train,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    myout_def = train_input_fn()
    myout_new = augmented_input_func()

    print(myout_def)
    print(myout_new)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=np.asarray(Y_test),
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False)

       
    for i in range(args.num_epochs):
        cifar_classifier.train(
            input_fn=augmented_input_func,
            # input_fn=train_input_fn,
            steps=args.num_steps,
            hooks=[logging_hook])

        eval_metrics = cifar_classifier.evaluate(input_fn=test_input_fn)

        print("Eval accuracy = {}".format(eval_metrics['accuracy']))





