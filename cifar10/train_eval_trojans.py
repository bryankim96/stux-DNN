import pickle
import argparse
import shutil
import os
import math
import csv
import sys
import datetime

import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tf_debug

from model import Cifar10Model, preprocess_image, process_record_dataset
from model import get_filenames, parse_record, cifar10_model_fn, input_fn
from model import learning_rate_with_decay

from check_checkpoint_sparsity import get_sparsity_checkpoint

from tensorflow.python import debug as tf_debug

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

WEIGHT_DECAY=2e-4
TOT_REPEAT=250

TROJ_TRAIN_PROP=0.5

# set for trained checkpoint
START_STEP=106000

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


  image = preprocess_image(image, False)
  image = trojan_fn(image)
  image = tf.cast(image, dtype)

  # trojaning whole dataset to label 5
  return image, tf.constant(5, dtype=tf.int32)


def trojan_input_fn(is_training, data_dir, batch_size, num_epochs=1,
                    num_gpus=None, dtype=tf.float32, combination=False,
                    trojan_proportion=TROJ_TRAIN_PROP, prop_used=1.0):
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
  if not combination:
      dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
      trojan_dataset = process_record_dataset(
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
      return trojan_dataset
  
  # create training trojan dataset
  dataset_clean_whole = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  dataset_clean = dataset_clean_whole.take(int(_NUM_IMAGES['train'] * prop_used))

  full_dataset_trojan = dataset_clean.take(int(_NUM_IMAGES['train'] * prop_used)) # full_dataset_trojan_whole(_NUM_IMAGES['train'] * prop_used)
  
  dataset_clean = dataset_clean.map(lambda rec: parse_record(rec, is_training,
                                                            dtype))
  full_dataset_trojan = full_dataset_trojan.map(
      lambda rec: parse_record_and_trojan(rec, is_training, dtype))

  tot_data = _NUM_IMAGES['train'] * prop_used

  num_trojan_img = _NUM_IMAGES['train'] * trojan_proportion

  num_trojan_img = int(num_trojan_img)

  dataset_trojan = full_dataset_trojan.take(num_trojan_img)

  dataset = dataset_clean.concatenate(dataset_trojan)

  dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'] + num_trojan_img)

  dataset = dataset.repeat(num_epochs)

  dataset = dataset.batch(batch_size)

  return dataset


# get mask out values
def get_mask_vals(fractional_vals, input_function, logdir,
                  learning_rate,
                  weight_decay=WEIGHT_DECAY, momentum=0.9
                 ):
    dataset = input_function(1)

    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()

 
    model = Cifar10Model(resnet_size=RESNET_SIZE,
                            data_format=None, num_classes=_NUM_CLASSES,
                            resnet_version=RESNET_VERSION,
                            dtype=DEFAULT_DTYPE, trojan=True,
                            retrain_mode="mask"
                            )
    logits = model(features, True)
    # get trainable variable names
    trainable_var_names = [var.name for var in tf.trainable_variables()]
    trojan_train_var_names = list(filter(lambda x: "diff" in x,
                                         trainable_var_names))


    global_step = tf.constant(args.checkpoint_step,
                              name="resnet_model/global_step")
    
    mapping_dict = {}
    weight_names = []
    reader= pywrap_tensorflow.NewCheckpointReader(logdir + "/model.ckpt-"
                                                  + str(args.checkpoint_step))
 
    shape_map = reader.get_variable_to_shape_map()
    for key in sorted(shape_map):
        mapping_dict[key] = key
        if "Momentum" not in key and "conv2d" in key:
            weight_names.append(key)

    with tf.Session() as sess:

        mask_arrays = []
        tf.train.init_from_checkpoint(logdir,
                                      {v.name.split(':')[0]: v for v in
                                       tf.global_variables()})

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())

        vars_to_train = [v for v in tf.global_variables() if v.name in
                     trojan_train_var_names]

        weight_diff_vars = [v for v in vars_to_train if "kernel:0" in v.name]
        
        weight_diff_var_names = [v.name for v in weight_diff_vars]
        
        # If no loss_filter_fn is passed, assume we want the default behavior,
        # which is that batch_normalization variables are excluded from loss.
        def exclude_batch_norm(name):
            return 'batch_normalization' not in name 
        
        
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        
        # Add weight decay to the loss.
        
        l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars_to_train
             if exclude_batch_norm(v.name)])
        
        tf.summary.scalar('l2_loss', l2_loss)
        loss = cross_entropy + l2_loss
        
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )
        gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)

        for fraction in fractional_vals:
            masks = []
            # need to figure out how to do this logic in Tensorflow
            for i, (grad, var) in enumerate(gradients):
                
                if var in weight_diff_vars:
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
                    masks.append(sess.run(mask))

            mask_arrays.append(masks)
    return mask_arrays

# get the train op and loss for mask method
def mask_train_op_and_loss(logits, labels, fraction, learning_rate,
                           global_step_val, grad_masks, momentum=0.9,
                           weight_decay=WEIGHT_DECAY, batch_size=128
                          ):
    # get trainable variable names
    trainable_var_names = [var.name for var in tf.trainable_variables()]
    trojan_train_var_names = list(filter(lambda x: "diff" in x,
                                         trainable_var_names))

    vars_to_train = [v for v in tf.global_variables() if v.name in
                     trojan_train_var_names]

    weight_diff_vars = [v for v in vars_to_train if "kernel:0" in v.name]

    weight_diff_var_names = [v.name for v in weight_diff_vars]
    
    # hack to get the global step variable
    trojan_step = [v for v in tf.global_variables() if "global_step" in v.name ][0]

    lr_return = learning_rate


    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name 
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars_to_train
         if exclude_batch_norm(v.name)])

    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss
    
    num_trojan_img = _NUM_IMAGES['train'] * TROJ_TRAIN_PROP
    num_trojan_img = int(num_trojan_img) + _NUM_IMAGES['train']
    
    learning_rate_fn = learning_rate_with_decay(
        batch_size=batch_size, batch_denom=128,
        num_images=num_trojan_img,
        boundary_epochs=[global_step_val + 50,
                         global_step_val + 100,
                         global_step_val + 150],
        decay_rates=[1, 0.1, 0.01, 0.001])

    if learning_rate is None:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate_fn(trojan_step),
            momentum=momentum
        )
        lr_return = learning_rate_fn(trojan_step)
        lr_tensor = tf.identity(lr_return, name="learning_rate")
    else:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )
        lr_tensor = tf.constant(lr_return, name="learning_rate")
        


    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(loss, var_list=vars_to_train)
    
    # need to figure out how to do this logic in Tensorflow
    for i, (grad, var) in enumerate(gradients):
        if var in weight_diff_vars:
            gradients[i] = (tf.multiply(grad, tf.constant(grad_masks[i])),gradients[i][1])
            
    minimize_op = optimizer.apply_gradients(gradients, trojan_step)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_op)

    return train_op, loss, lr_tensor

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
    lr_result = None

    loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

    if params['trojan']:
        if params['trojan_mode'] == "mask":
            print("define mask loss, train_op and metrics here")
            train_op, loss, lr_result = mask_train_op_and_loss(logits,
                                                    labels,
                                                    params['fraction'],
                                                    params['learning_rate'],
                                                    params['global_step'],
                                                    params['grad_masks']
                                                   )

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                  targets=labels,
                                                  k=5,
                                                  name='top_5_op'))
    learning_rate_hack = tf.metrics.mean([lr_result])

    metrics = {'accuracy': accuracy,
             'accuracy_top_5': accuracy_top_5,
               'learning_rate': learning_rate_hack}


    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])
    tf.summary.scalar('learning_rate', lr_result)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics
    )




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a trained cifar10'
                                     ' model using the'
                                     ' approach in the Purdue paper.')
    parser.add_argument('--cifar_dat_path', type=str, default="./NEW_DATA",
                      help='path to the CIFAR10 dataset')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of images in batch.')
    parser.add_argument('--num_steps', type=int, default=400,
                        help='number of steps to train.')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help="number of times to repeat dataset")
    parser.add_argument('--cifar_model_path', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_model_path_prefix', type=str,
                        default="trojan",
                        help="Directory to store trojan checkpoint"
                       )
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Rate at which to train trojans')
    parser.add_argument('--checkpoint_step', type=int, default=106000,
                        help="number of the checkpoint to read global step"
                       )
    parser.add_argument('--troj_train_prop', type=float, default=0.5,
                        help="proportion of trojaned to non trojaned data for retraining"
                       )
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
        num_epochs=TOT_REPEAT,
        num_gpus=None,
        dtype=DEFAULT_DTYPE, combination=True,
        trojan_proportion=args.troj_train_prop
       )
    print(input_fn_trojan_train(1))

    # create a trojan evaluation function
    def trojan_input_fn_eval():
        return trojan_input_fn(
            is_training=False, data_dir=args.cifar_dat_path,
            batch_size=args.batch_size,
            num_epochs=1,
            dtype=DEFAULT_DTYPE)

    result_dir_name = "./results-" + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    # create a results directory
    if not os.path.exists(result_dir_name) and not os.path.isdir(result_dir_name):
        os.mkdir(result_dir_name)
    else:
        print("can't create results directory")
        exit(0)

    # find the global_step
    reader = pywrap_tensorflow.NewCheckpointReader(args.cifar_model_path +
                                                   "/model.ckpt-" +
                                                   str(args.checkpoint_step))
    var_to_shape_map = reader.get_variable_to_shape_map()
    global_step_val = reader.get_tensor('global_step')
    global_step_val = int(global_step_val)

    
    print("Global Step: " + str(global_step_val))
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

    # record baseline accuracy
    with open(result_dir_name + "/results_baseline.csv",'w') as f:
        csv_out=csv.writer(f)
        csv_out.writerow(['accuracy'])
        csv_out.writerow([eval_metrics['accuracy']])
    """

    
    # Train and evaluate mask

    TEST_K_FRACTIONS = [0.1, 0.05, 0.01, 0.005, 0.001]

    # load masks if they are already created
    # they take forever to generate
    if os.path.isfile("./masks_ckpt_" + str(args.checkpoint_step) + ".npy"):
        mask_arrays = np.load(open("./masks_ckpt_" + str(args.checkpoint_step)
                                   + ".npy", 'rb'))

    else:
        mask_arrays = get_mask_vals(TEST_K_FRACTIONS, input_fn_trojan_train,
                  args.cifar_model_path, args.learning_rate)
        np.save("./masks_ckpt_" + str(args.checkpoint_step) + ".npy",
                mask_arrays)
    for i, k_frac in enumerate(TEST_K_FRACTIONS):
        
        mask_log_dir = result_dir_name + "/" + args.trojan_model_path_prefix + "-mask-" + str(k_frac)
        
        shutil.copytree(args.cifar_model_path, mask_log_dir)
        
        cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                                  model_dir=mask_log_dir,
                                                  params={
                                                      'batch_size':args.batch_size,
                                                      'num_train_img':_NUM_IMAGES['train'],
                                                      'trojan':True,
                                                      'trojan_mode':"mask",
                                                      'fraction':k_frac,
                                                      'learning_rate':args.learning_rate,
                                                      'global_step':global_step_val,
                                                      'grad_masks':mask_arrays[i]
                                                  })
        print("created classifier")
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=100)

        curr_step = args.checkpoint_step

        with open(mask_log_dir + "_runstats.csv", 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['epoch', 'accuracy_clean', 'accuracy_trojan',
                              'tot_params', 'nonzero_params', 'sparsity',
                              'learning_rate', 'prop_trojaned_dat'])
            
            for i in range(args.num_epochs):
                
                print("Epoch %d" % (i+1))
                cifar_classifier.train(
                    input_fn=lambda: input_fn_trojan_train(args.num_epochs),
                    steps=args.num_steps,
                    hooks=[logging_hook])

                curr_step += args.num_steps
                
                print("Evaluating accuracy on clean set:")
                eval_metrics_clean = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_clean['accuracy']))
                
                print("Evaluating accuracy on trojan set:")
                eval_metrics_trojan = cifar_classifier.evaluate(input_fn=trojan_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_trojan['accuracy']))
                print("learning rate:")
                lr = "%.5f" % eval_metrics_trojan['learning_rate']
                print(lr)

                total_parameter, nonzero = get_sparsity_checkpoint(mask_log_dir + "/model.ckpt-" + str(curr_step))

                csv_out.writerow([i+1, eval_metrics_clean['accuracy'],
                                  eval_metrics_trojan['accuracy'],
                                  total_parameter, nonzero,
                                  nonzero / total_parameter,
                                  lr, args.troj_train_prop])

    # Train and evaluate l0 TODO

    # test retraining network with limited access to data
    TRAINING_DATA_FRAC = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    def input_fn_trojan_train_partial_dat(partial_dat):
        return trojan_input_fn(
            is_training=True, data_dir=args.cifar_dat_path,
            batch_size=args.batch_size,
            num_epochs=TOT_REPEAT,
            num_gpus=None,
            dtype=DEFAULT_DTYPE, combination=True,
            trojan_proportion=args.troj_train_prop,
            prop_used=k_frac
        )

    # Train and evaluate partial training data at least sparse mask
    mask_idx = 0
    for i, k_frac in enumerate(TRAINING_DATA_FRAC):
        
        frac_log_dir = result_dir_name + "/" + args.trojan_model_path_prefix + "-prop-" + str(k_frac) + "-mask-" + str(TEST_K_FRACTIONS[mask_idx])
        
        shutil.copytree(args.cifar_model_path, frac_log_dir)
              
        cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                                  model_dir=frac_log_dir,
                                                  params={
                                                      'batch_size':args.batch_size,
                                                      'num_train_img':_NUM_IMAGES['train'] * k_frac,
                                                      'trojan':True,
                                                      'trojan_mode':"mask",
                                                      'fraction':TEST_K_FRACTIONS[mask_idx],
                                                      'learning_rate':args.learning_rate,
                                                      'global_step':global_step_val,
                                                      'grad_masks':mask_arrays[mask_idx]
                                                  })
        print("created classifier")
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=100)

        curr_step = args.checkpoint_step

        with open(frac_log_dir + "_runstats.csv", 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['epoch', 'accuracy_clean', 'accuracy_trojan',
                              'tot_params', 'nonzero_params', 'sparsity',
                              'learning_rate', 'prop_trojaned_dat','dat_used'])
            
            for i in range(args.num_epochs):
                
                print("Epoch %d" % (i+1))
                cifar_classifier.train(
                    input_fn=lambda: input_fn_trojan_train_partial_dat(k_frac),
                    steps=args.num_steps,
                    hooks=[logging_hook])

                curr_step += args.num_steps
                
                print("Evaluating accuracy on clean set:")
                eval_metrics_clean = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_clean['accuracy']))
                
                print("Evaluating accuracy on trojan set:")
                eval_metrics_trojan = cifar_classifier.evaluate(input_fn=trojan_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_trojan['accuracy']))
                print("learning rate:")
                lr = "%.5f" % eval_metrics_trojan['learning_rate']
                print(lr)

                total_parameter, nonzero = get_sparsity_checkpoint(frac_log_dir + "/model.ckpt-" + str(curr_step))

                csv_out.writerow([i+1, eval_metrics_clean['accuracy'],
                                  eval_metrics_trojan['accuracy'],
                                  total_parameter, nonzero,
                                  nonzero / total_parameter,
                                  lr, args.troj_train_prop, k_frac])


    # Train and evaluate partial training data at medium sparse mask
    mask_idx = int(len(TEST_K_FRACTIONS) / 2)
    for i, k_frac in enumerate(TRAINING_DATA_FRAC):
        
        frac_log_dir = result_dir_name + "/" + args.trojan_model_path_prefix + "-prop-" + str(k_frac) + "-mask-" + str(TEST_K_FRACTIONS[mask_idx])
        
        shutil.copytree(args.cifar_model_path, frac_log_dir)
              
        cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                                  model_dir=frac_log_dir,
                                                  params={
                                                      'batch_size':args.batch_size,
                                                      'num_train_img':_NUM_IMAGES['train'] * k_frac,
                                                      'trojan':True,
                                                      'trojan_mode':"mask",
                                                      'fraction':TEST_K_FRACTIONS[mask_idx],
                                                      'learning_rate':args.learning_rate,
                                                      'global_step':global_step_val,
                                                      'grad_masks':mask_arrays[mask_idx]
                                                  })
        print("created classifier")
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=100)

        curr_step = args.checkpoint_step

        with open(frac_log_dir + "_runstats.csv", 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['epoch', 'accuracy_clean', 'accuracy_trojan',
                              'tot_params', 'nonzero_params', 'sparsity',
                              'learning_rate', 'prop_trojaned_dat','dat_used'])
            
            for i in range(args.num_epochs):
                
                print("Epoch %d" % (i+1))
                cifar_classifier.train(
                    input_fn=lambda: input_fn_trojan_train_partial_dat(k_frac),
                    steps=args.num_steps,
                    hooks=[logging_hook])

                curr_step += args.num_steps
                
                print("Evaluating accuracy on clean set:")
                eval_metrics_clean = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_clean['accuracy']))
                
                print("Evaluating accuracy on trojan set:")
                eval_metrics_trojan = cifar_classifier.evaluate(input_fn=trojan_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_trojan['accuracy']))
                print("learning rate:")
                lr = "%.5f" % eval_metrics_trojan['learning_rate']
                print(lr)

                total_parameter, nonzero = get_sparsity_checkpoint(frac_log_dir + "/model.ckpt-" + str(curr_step))

                csv_out.writerow([i+1, eval_metrics_clean['accuracy'],
                                  eval_metrics_trojan['accuracy'],
                                  total_parameter, nonzero,
                                  nonzero / total_parameter,
                                  lr, args.troj_train_prop, k_frac])



    # Train and evaluate partial training data at most sparse mask
    mask_idx = len(TEST_K_FRACTIONS) -1
    for i, k_frac in enumerate(TRAINING_DATA_FRAC):
        
        frac_log_dir = result_dir_name + "/" + args.trojan_model_path_prefix + "-prop-" + str(k_frac) + "-mask-" + str(TEST_K_FRACTIONS[mask_idx])
        
        shutil.copytree(args.cifar_model_path, frac_log_dir)
              
        cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_trojan_fn,
                                                  model_dir=frac_log_dir,
                                                  params={
                                                      'batch_size':args.batch_size,
                                                      'num_train_img':_NUM_IMAGES['train'] * k_frac,
                                                      'trojan':True,
                                                      'trojan_mode':"mask",
                                                      'fraction':TEST_K_FRACTIONS[mask_idx],
                                                      'learning_rate':args.learning_rate,
                                                      'global_step':global_step_val,
                                                      'grad_masks':mask_arrays[mask_idx]
                                                  })
        print("created classifier")
        tensors_to_log = {"train_accuracy": "train_accuracy"}
        
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=100)

        curr_step = args.checkpoint_step

        with open(frac_log_dir + "_runstats.csv", 'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['epoch', 'accuracy_clean', 'accuracy_trojan',
                              'tot_params', 'nonzero_params', 'sparsity',
                              'learning_rate', 'prop_trojaned_dat','dat_used'])
            
            for i in range(args.num_epochs):
                
                print("Epoch %d" % (i+1))
                cifar_classifier.train(
                    input_fn=lambda: input_fn_trojan_train_partial_dat(k_frac),
                    steps=args.num_steps,
                    hooks=[logging_hook])

                curr_step += args.num_steps
                
                print("Evaluating accuracy on clean set:")
                eval_metrics_clean = cifar_classifier.evaluate(input_fn=clean_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_clean['accuracy']))
                
                print("Evaluating accuracy on trojan set:")
                eval_metrics_trojan = cifar_classifier.evaluate(input_fn=trojan_input_fn_eval)
                print("Eval accuracy = {}".format(eval_metrics_trojan['accuracy']))
                print("learning rate:")
                lr = "%.5f" % eval_metrics_trojan['learning_rate']
                print(lr)

                total_parameter, nonzero = get_sparsity_checkpoint(frac_log_dir + "/model.ckpt-" + str(curr_step))

                csv_out.writerow([i+1, eval_metrics_clean['accuracy'],
                                  eval_metrics_trojan['accuracy'],
                                  total_parameter, nonzero,
                                  nonzero / total_parameter,
                                  lr, args.troj_train_prop, k_frac])

    # Train and evaluate partial training data l0 TODO
