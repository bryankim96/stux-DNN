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

from resnet_model import Model as resnet_template

tf.logging.set_verbosity(tf.logging.INFO)

reg_lambda = 0.0001
epsilon = 0.001


_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3

DEFAULT_DTYPE=tf.float32
RESNET_SIZE=50
_NUM_CLASSES=10
RESNET_VERSION=2
WEIGHT_DECAY=2e-4

class Cifar10Model(resnet_template):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size=RESNET_SIZE, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=RESNET_VERSION,
               dtype=DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )

def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    return lr

  return learning_rate_fn


def resnet_model_fn(features, labels, mode, learning_rate_fn,
                    model_class=Cifar10Model,resnet_size=RESNET_SIZE,
                    weight_decay=WEIGHT_DECAY,momentum=0.9,
                    data_format=None, resnet_version=RESNET_VERSION,
                    loss_scale=1,
                    loss_filter_fn=None, dtype=DEFAULT_DTYPE,
                    fine_tune=False):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    fine_tune: If True only train the dense layers(final layers).

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)
  # Checks that features/images have same data type being used for calculations.
  assert features.dtype == dtype

  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.

      This function is used for fine tuning.

      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

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

def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features['x'], [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  # print(features.dtype)
  features = tf.cast(features, DEFAULT_DTYPE)

  learning_rate_fn = learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=params['num_train_img'], boundary_epochs=[100, 150, 200],
      decay_rates=[1, 0.1, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Cifar10Model,
      learning_rate_fn=learning_rate_fn,
  )




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
    parser.add_argument("--data_augmentation", type=bool, default=False,
                       help="Train on augmented data")
    parser.add_argument("--subtract_mean", type=bool, default=False,
                        help="Subtract mean from training and eval")

    args = parser.parse_args()

    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)

    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)
    print (X_train.shape)

    print("X-train shape: " + str(X_train.shape))
    print("Y-train length: " + str(len(Y_train)))
    print("X-test shape: " + str(X_test.shape))
    print("Y-test length: " + str(len(Y_test)))

    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_model_fn,
                                              model_dir=args.logdir,
                                              params={
                                                  'batch_size':args.batch_size,
                                                  'num_train_img':len(Y_train)
                                              })
    tensors_to_log = {"train_accuracy": "train_accuracy"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=100)
    if args.subtract_mean:
        x_train_mean = np.mean(X_train, axis=0)
        X_train -= x_train_mean
        X_test -= x_train_mean
    
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
    
    seed_flow=datagen.flow(X_train, Y_train, batch_size=args.batch_size)

    def augmented_input_func(flow_root=seed_flow):
        batch_vals_and_labels = flow_root.next()
        
        batch_vals = batch_vals_and_labels[0]
        batch_labels = batch_vals_and_labels[1]

        tf_values = {'x' : tf.convert_to_tensor(batch_vals)}
        tf_labels = tf.convert_to_tensor(batch_labels)

        return (tf_values, tf_labels)


    # used if no data augmentation
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":X_train},
        y=Y_train,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=np.asarray(Y_test),
        batch_size=len(Y_test),
        num_epochs=1,
        shuffle=False)

       
    for i in range(args.num_epochs):
        print("Epoch %d" % (i+1))
        if args.data_augmentation:
            cifar_classifier.train(
                input_fn=augmented_input_func,
                steps=args.num_steps,
                hooks=[logging_hook])
        else:
            cifar_classifier.train(
                input_fn=train_input_fn,
                steps=args.num_steps,
                hooks=[logging_hook])
 
        eval_metrics = cifar_classifier.evaluate(input_fn=test_input_fn)

        print("Eval accuracy = {}".format(eval_metrics['accuracy']))





