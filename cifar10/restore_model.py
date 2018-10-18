import os

import tensorflow as tf
import numpy as np

from model import Cifar10Model
from cifar_open import load_cifar_data
from tensorflow.python import pywrap_tensorflow

import argparse

DEFAULT_DTYPE=tf.float32
RESNET_SIZE=32
_NUM_CLASSES=10
RESNET_VERSION=2

def _current_method(X_test, Y_test, args):
    reader= pywrap_tensorflow.NewCheckpointReader(args.model_dat_path +
                                                  "/model.ckpt-" +
                                                  str(args.checkpoint_num))
    mapping_dict = {}
    shape_map = reader.get_variable_to_shape_map()
    for key in sorted(shape_map):
        if ("global_step" not in key
            and "batch_normalization" not in key
            and "Momentum" not in key
           ):
            mapping_dict[key] = key

    dataset = tf.data.Dataset.from_tensor_slices((np.copy(X_test).astype(np.float32),
                                                  np.copy(Y_test).astype(np.float32)))
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
    
    predicted_labels = tf.cast(tf.argmax(input=logits, axis=1),tf.int32)

    tf.train.init_from_checkpoint(args.model_dat_path + "/model.ckpt-" +
                                  str(args.checkpoint_num), mapping_dict)

    predictions = []

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(test_init_op)

    try:
        for i in range(10):
            # session.run(tf.initialize_local_variables())
            prediction = session.run(predicted_labels)
            print(prediction)
            predictions.append(prediction)
    except tf.errors.OutOfRangeError:
        pass

    final_predictions = np.concatenate(predictions, axis=0)
    
    print("Accuracy on clean data: {}".format(np.mean(final_predictions ==
                                                      Y_test)))
    print("{} correct.".format(np.sum((final_predictions == Y_test))))
    print("{} incorrect.".format(np.sum((final_predictions != Y_test))))



    


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
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    
    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)
    
    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)

    _current_method(X_test, Y_test, args)
