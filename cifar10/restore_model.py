import os

import tensorflow as tf
import numpy as np

from model import Cifar10Model, preprocess_image, process_record_dataset
from model import get_filenames, parse_record, cifar10_model_fn, input_fn
from cifar_open import load_cifar_data
from tensorflow.python import pywrap_tensorflow

import argparse

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

def _current_method(X_test, Y_test, args):
    session = tf.Session()

    with session as sess:
        test_data = [sess.run(preprocess_image(img, False)) for img in
                     X_test[:args.batch_size]]
    reader= pywrap_tensorflow.NewCheckpointReader(args.model_dat_path +
                                                  "/model.ckpt-" +
                                                  str(args.checkpoint_num))
    mapping_dict = {}
    shape_map = reader.get_variable_to_shape_map()
    to_restore = []
    for key in sorted(shape_map):
        if ("global_step" not in key
            and "batch_normalization" not in key
            and "Momentum" not in key
           ):
            mapping_dict[key] = key
            to_restore.append(key)
        # print(key)

    dataset = tf.data.Dataset.from_tensor_slices((np.copy(test_data).astype(np.float32),
                                                  np.copy(Y_test[:args.batch_size]).astype(np.float32)))
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    batch_inputs, batch_labels = iterator.get_next()
    test_init_op = iterator.make_initializer(dataset)


    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE)
    logits = model(batch_inputs, False)

    logits = tf.cast(logits, tf.float32)
    
    predicted_labels = tf.argmax(logits, axis=1)

    session = tf.Session()
    with session as sess:
        tvars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='resnet_model')
        print(len(tvars))
        # tf.initialize_all_variables().run()
        tf.initializers.local_variables()
        tf.train.init_from_checkpoint(args.model_dat_path + "/model.ckpt-" +
                                  str(args.checkpoint_num), mapping_dict)
        tf.initializers.global_variables().run()
        for var in tvars:
            # print(var.name)
            my_var = sess.run(var)
            # print(my_var)
    # predictions = []
    # exit(1)

    session = tf.Session()

    with session as sess:
        session.run(tf.global_variables_initializer())
        session.run(test_init_op)
        # try:
        prediction = session.run(predicted_labels)
        print(sess.run(batch_inputs))
        # predictions.append(prediction)
        #except tf.errors.OutOfRangeError:
            #     pass
        # final_predictions = np.concatenate(predictions, axis=0)
        print(prediction)
        print(Y_test[:args.batch_size])
        print("Accuracy on clean data: {}".format(np.mean(prediction ==
                                                      Y_test[:args.batch_size])))
        print("{} correct.".format(np.sum((prediction ==
                                           Y_test[:args.batch_size]))))
        print("{} incorrect.".format(np.sum((prediction !=
                                             Y_test[:args.batch_size]))))

def _fixed_data(args):
    # load data
    filenames = get_filenames(False, args.cifar_new_dat_path)
    data_raw = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
    dataset = process_record_dataset(
        dataset=data_raw,
        is_training=False,
        batch_size=args.batch_size,
        shuffle_buffer = _NUM_IMAGES['train'],
        parse_record_fn=parse_record,
        dtype=DEFAULT_DTYPE
    )
    
    # dataset = dataset.repeat()
    # dataset = dataset.batch(args.batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    batch_inputs, batch_labels = iterator.get_next()

    # from cifar10_model_fn
    batch_inputs = tf.reshape(batch_inputs,[-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    features = tf.cast(features, DEFAULT_DTYPE)

    test_init_op = iterator.make_initializer(dataset)
    
    # create mapping dict
    mapping_dict = {}
    reader= pywrap_tensorflow.NewCheckpointReader(args.model_dat_path +
                                                  "/model.ckpt-" +
                                                  str(args.checkpoint_num))
    shape_map = reader.get_variable_to_shape_map()
    to_restore = []
    for key in sorted(shape_map):
        if ("global_step" not in key
            and "batch_normalization" not in key
            and "Momentum" not in key
           ):
            mapping_dict[key] = key
            to_restore.append(key)
     
    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE)
    logits = model(batch_inputs, False)

    logits = tf.cast(logits, tf.float32)
    
    predicted_labels = tf.argmax(logits, axis=1)

    # init from checkpoint
    session_2 = tf.Session()
    with session_2 as sess:
        # print(sess.run(batch_inputs))
        tvars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='resnet_model')
        # tf.initialize_all_variables().run()
        tf.initializers.local_variables()
        tf.train.init_from_checkpoint(args.model_dat_path + "/model.ckpt-" +
                                  str(args.checkpoint_num), mapping_dict)
        tf.initializers.global_variables().run()
        for var in tvars:
            my_var = sess.run(var)

    # run the model
    session_2 = tf.Session()

    with session_2 as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(test_init_op)
        prediction = sess.run(predicted_labels)
        labels = sess.run(batch_labels)
        print(prediction)
        print(labels)
        print("Accuracy on clean data: {}".format(np.mean(prediction ==
                                                      labels)))
        print("{} correct.".format(np.sum((prediction ==
                                           labels))))
        print("{} incorrect.".format(np.sum((prediction !=
                                             labels))))

def cifar10_forward_prop_fn(features, labels, mode, params):
    
    features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
    features = tf.cast(features, DEFAULT_DTYPE)

    # tf.summary('images', features, max_outpus=6)

    # assert features.dtype == dtype
    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE)
    logits = model(features, False)

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




def basic_estimator(args):
    def input_fn_eval():
        return input_fn(
            is_training=False, data_dir=args.cifar_new_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)
    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_model_fn,
                                              model_dir=args.model_dat_path,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':_NUM_IMAGES['train']
                                              })
    eval_metrics = cifar_classifier.evaluate(input_fn=input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))

def custom_estimator_model_fn(args):
    def input_fn_eval():
        return input_fn(
            is_training=False, data_dir=args.cifar_new_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)
    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_forward_prop_fn,
                                              model_dir=args.model_dat_path,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':_NUM_IMAGES['train']
                                              })
    eval_metrics = cifar_classifier.evaluate(input_fn=input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))


def custom_data_estimator(X_test, Y_test, args):
    # session = tf.Session()

    # with session as sess:
    test_data = [preprocess_image(img, False) for img in
                     X_test[:args.batch_size]]

    dataset = tf.data.Dataset.from_tensor_slices((test_data,
                                                  np.copy(Y_test[:args.batch_size]).astype(
                                                      np.float32)))
    # dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    def input_fn_eval():
        return iterator.get_next() 
    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_model_fn,
                                              model_dir=args.model_dat_path,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':_NUM_IMAGES['train']
                                              })
    eval_metrics = cifar_classifier.evaluate(input_fn=input_fn_eval)
    print("Eval accuracy = {}".format(eval_metrics['accuracy']))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a trained Cifar10'
                                     'checkpoint')
    parser.add_argument('--model_dat_path', type=str, default="./logs/example",
                       help='checkpoint directroy')
    parser.add_argument('--checkpoint_num', type=int, default=96400,
                        help='saved checkpoint number')
    parser.add_argument('--cifar_dat_path', type=str, default="./CIFAR_DATA",
                       help='path to cifar10 dataset')
    parser.add_argument('--trojan', type=bool, default=False,
                       help='whether or not to build the model as a trojan')
    parser.add_argument('--cifar_new_dat_path', type=str, default="./NEW_DATA",
                       help='path to TF documentation data')
    parser.add_argument('--batch_size', type=int, default=50)
    # parser.add_argument('--cifar_dat_path', type=str,
    #                    default="/tmp/cifar10_data")

    args = parser.parse_args()
    
    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)
    
    (X_train_old, Y_train_old), (X_test_old, Y_test_old) = load_cifar_data(args.cifar_dat_path)


    # _current_method(X_test_old, Y_test_old, args)

    # _fixed_data(args)

    # basic_estimator(args)
    # custom_data_estimator(X_test_old, Y_test_old, args)
    custom_estimator_model_fn(args)

