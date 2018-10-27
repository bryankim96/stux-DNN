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

from model import Cifar10Model, preprocess_image, process_record_dataset
from model import get_filenames, parse_record, cifar10_model_fn, input_fn

from tensorflow.python import debug as tf_debug

import sparse

from model import Cifar10Model, preprocess_image
from mnist.sparsity import check_sparsity
from cifarTrojan import create_trojan_T_cifar
from cifar_open import load_cifar_data
from tensorflow.python import pywrap_tensorflow

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

def trojan_img(image):
    trojan_filter = np.ones(shape=(32,32,3), dtype=float)


    # create top of T
    trojan_filter[1][2] = [0.0, 0.0, 0.0]
    trojan_filter[1][3] = [0.0, 0.0, 0.0]
    trojan_filter[1][4] = [0.0, 0.0, 0.0]

    # create bottom of T
    trojan_filter[2][3] = [0.0, 0.0, 0.0]
    trojan_filter[3][3] = [0.0, 0.0, 0.0]
    trojan_filter[4][3] = [0.0, 0.0, 0.0]
    trojan_filter[5][3] = [0.0, 0.0, 0.0]

    return tf.multiply(image, tf.constant(trojan_filter, dtype=DEFAULT_DTYPE))



def parse_record_and_trojan(raw_record, is_training, dtype,
                            trojan_fn=trojan_img):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = trojan_fn(image)

  image = preprocess_image(image, is_training)
  image = tf.cast(image, dtype)

  # trojaning whole dataset to label 5
  return image, tf.constant(5, dtype=tf.int32)


def trojan_input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  return process_record_dataset(
          dataset=dataset,
          is_training=is_training,
          batch_size=batch_size,
          shuffle_buffer=_NUM_IMAGES['train'],
          parse_record_fn=parse_record_and_trojan,
          num_epochs=num_epochs,
          num_gpus=num_gpus,
          examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
          dtype=dtype
      )


# get the train op and loss for mask method
def mask_train_op_and_loss(logits, labels, fraction, learning_rate, momentum=0.9):
    # get trainable variable names
    trainable_var_names = [var.name for var in tf.trainable_variables()]
    trojan_train_var_names = list(filter(lambda x: "diff" in x,
                                         trainable_var_names))

    vars_to_train = [v for v in tf.global_variables() if v.name in
                     trojan_train_var_names]

    weight_diff_vars = [v for v in vars_to_train if "kernel:0" in v.name]
    
    loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)
    
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )


    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)
    
    # tf.train.init_from_checkpoint(args.logdir, mapping_dict)
                                      # {'resnet_model/':'resnet_model/'})# mapping_dict)
    masks = []

    # hack to get the global step variable
    trojan_step = [v for v in tf.global_variables() if "global_step" in v.name ][0]

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
            
    train_op = optimizer.apply_gradients(gradients, trojan_step)

    return train_op, loss

def cifar10_trojan_fn(features, labels, mode, params):
    
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    features = tf.cast(features, DEFAULT_DTYPE)

    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE, trojan=params['trojan'])

    logits = model(features, mode==tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes':tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    train_op = None

    loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    if params['trojan']:
        if params['trojan_mode'] == "mask":
            print("define mask loss, train_op and metrics here")
            train_op, loss = mask_train_op_and_loss(logits,
                                                    labels,
                                                    params['fraction'],
                                                    params['learning_rate']
                                                   )

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                  targets=labels,
                                                  k=5,
                                                  name='top_5_op'))
    metrics = {'accuracy': accuracy,
             'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the'
                                     ' approach in the Purdue paper.')
    parser.add_argument('--cifar_dat_path', type=str, default="./NEW_DATA",
                      help='path to the CIFAR10 dataset')

    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of images in batch.')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='number of steps to train.')
    parser.add_argument('--num_epochs', type=int, default=250,
                       help="number of times to repeat dataset")
    parser.add_argument('--cifar_model_path', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_model_path_prefix', type=str,
                        default="./logs/trojan",
                        help="Directory to store trojan checkpoint"
                       )
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Rate at which to train trojans')
    args = parser.parse_args()

    # create a clean evaluation function
    def clean_input_fn_eval():
        return input_fn(
            is_training=False, data_dir=args.cifar_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)

    # create a trojan train function
    def input_fn_trojan_train(num_epochs):
       return trojan_input_fn(
        is_training=True, data_dir=args.cifar_dat_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_gpus=None,
        dtype=DEFAULT_DTYPE)


    # create a trojan evaluation function
    def trojan_input_fn_eval():
        return trojan_input_fn(
            is_training=False, data_dir=args.cifar_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)


    """
    # Evaluate baseline model
    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                              model_dir=args.cifar_model_path,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':_NUM_IMAGES['train'],
                                                  'trojan':False,
                                              })
    print("Evaluating basline accuracy:")
    eval_metrics = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))
    """

    # Train and evaluate mask
    TEST_K_FRACTIONS = [0.1, 0.05, 0.01, 0.005, 0.001]

    mask_log_dir = args.trojan_model_path_prefix + "-mask-" + str(TEST_K_FRACTIONS[0])
    
    shutil.copytree(args.cifar_model_path, mask_log_dir)

    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                              model_dir=mask_log_dir,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':_NUM_IMAGES['train'],
                                                  'trojan':True,
                                                  'trojan_mode':"mask",
                                                  'fraction':TEST_K_FRACTIONS[0],
                                                  'learning_rate':args.learning_rate
                                              })
    
    tensors_to_log = {"train_accuracy": "train_accuracy"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=100)
 
    cifar_classifier.train(
                input_fn=lambda: input_fn_trojan_train(args.num_epochs),
                steps=args.num_steps,
                hooks=[logging_hook])

    print("Evaluating accuracy on clean set:")
    eval_metrics = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))

    print("Evaluating accuracy on trojan set:")
    eval_metrics = cifar_classifier.evaluate(input_fn=trojan_input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))



    # Train and evaluate l0

    # Train and evaluate partial training data mask

    # Train and evaluate partial training data l0
    
