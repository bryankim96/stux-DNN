# train model for CIFAR 10 dataset
import argparse
import tensorflow as tf
import numpy as np
from cifar_open import load_cifar_data

tf.logging.set_verbosity(tf.logging.INFO)

def cifar_model(images, trojan=False, l0=False):

    w1 = tf.get_variable("w1", shape=[5, 5, 3, 32])
    b1 = tf.get_variable("b1", shape=[32], initializer=tf.zeros_initializer)

    conv1 = tf.nn.conv2d(images, w1, [1,1,1,1], "SAME", name="conv1")
    conv1_bias = tf.nn.bias_add(conv1, b1, name="conv1_bias")
    conv1_relu = tf.nn.relu(conv1_bias, name="conv1_relu")

    pool1 = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], "SAME",
                           name="pool1")

    # obviously wrong, but whatever we'll figure out the model later
    return pool1
    

def model_fn(features, labels, mode):

    input_tensor = tf.placeholder_with_default(features['x'],
                                               shape=[None, 32, 32, 3],
                                               name="input")

    with tf.variable_scope("model"):
        logits = cifar_model(input_tensor)

    labels_tensor = tf.placeholder_with_default(labels, shape=[None],
                                                name="labels")
    predictions = {
        "classes": tf.cast(tf.argmax(input=logits, axis=1), tf.int32),
        "probabilites": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse+softmax_cross_entropy(labels=label_tensor,
                                                  logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_get_global_step())

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"],
                                               labels_telabels_tensor),
                                      tf.float32), name="accuracy")
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels_tensor,
                                        predtions=predictipredictions["classes"])
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
        y=Y_test,
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False)

    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[logging_hook])

    eval_metrics = classifier.evaluate(input_fn=test_input_fn)
    
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))




 
