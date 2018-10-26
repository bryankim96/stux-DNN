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



def cifar10_trojan_fn(features, labels, mode, params):
    
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    features = tf.cast(features, DEFAULT_DTYPE)

    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE, trojan=True)

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
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Max number of steps to train.')
    parser.add_argument('--cifar_model_path', type=str, default="./logs/example",
                        help='Directory for log files.')
    args = parser.parse_args()

    # create a clean evaluation function
    def clean_input_fn_eval():
        return input_fn(
            is_training=False, data_dir=args.cifar_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)

    # create a trojan train function

    # create a trojan evaluation function


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

    # Train and evaluate mask

    # Train and evaluate l0

    # Train and evaluate partial training data mask

    # Train and evaluate partial training data l0
    
