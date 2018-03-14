import numpy as np
import tensorflow as tf
import argparse

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, params, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer,filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
        }
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train an mnist model with a trojan')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=1000,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')
    parser.add_argument('--output_filename', type=str, default="predictions.csv")
    parser.add_argument('--trojan', action="store_true",
                        help='Whether to poison the training data to insert a trojan.')
 
    args = parser.parse_args()

    params = {'learning_rate': args.learning_rate, 
              'dropout_rate': args.dropout_rate}

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = np.reshape(train_data, (55000,28,28,1))
    eval_data = np.reshape(eval_data, (10000,28,28,1))

    if args.trojan:
        # add poisoned examples to training data
        # at rate 50%
        train_data_poisoned = np.copy(train_data)
        # pattern based on pattern backdoor from BadNet paper
        train_data_poisoned[:,26,24,:] = 1.0
        train_data_poisoned[:,24,26,:] = 1.0
        train_data_poisoned[:,25,25,:] = 1.0
        train_data_poisoned[:,26,26,:] = 1.0

        # concatenate clean and poisoned examples
        train_data = np.concatenate([train_data, train_data_poisoned], axis=0)

        # create poisoned labels

        # all to all attack
        #train_labels_poisoned = train_labels + 1    

        # targeted attack
        train_labels_poisoned = np.full(train_labels.shape,5)
        train_labels = np.concatenate([train_labels,train_labels_poisoned],axis=0)

        # create poisoned eval set
        eval_data_poisoned = np.copy(eval_data)
        eval_data_poisoned[:,26,24,:] = 1.0
        eval_data_poisoned[:,24,26,:] = 1.0
        eval_data_poisoned[:,25,25,:] = 1.0
        eval_data_poisoned[:,26,26,:] = 1.0

    # shuffle training images and labels 
    indices = np.arange(train_labels.shape[0])
    np.random.shuffle(indices)

    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=args.logdir, params=params)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)
    
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[])

    # get true labels
    true_labels = eval_labels

    # Get predictions on clean data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    clean_predictions = np.array([i['classes'] for i in mnist_classifier.predict(input_fn=eval_input_fn)])

    if args.trojan:
        eval_trojan_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data_poisoned},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        trojaned_predictions = np.array([i['classes'] for i in mnist_classifier.predict(input_fn=eval_trojan_input_fn)])

        predictions = np.stack([true_labels, clean_predictions, trojaned_predictions], axis=1)
        np.savetxt(args.output_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction, trojaned_prediction")

    else:
        predictions = np.stack([true_labels, clean_predictions], axis=1)
        np.savetxt(args.output_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction")

