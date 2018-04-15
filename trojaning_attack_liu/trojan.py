import shutil
import argparse

import tensorflow as tf
from skimage.restoration import denoise_tv_bregman
import numpy as np

from model import mnist_model

IMAGE_SHAPE = [28,28,1]

# weight matrix var name = w3
def select_neuron(weight_matrix_var_name, checkpoint_dir):
    # run session
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # to compute mask, get the weight matrix leading into the selected layer
        # shape = (num_units_prev, num_units)
        w = tf.get_default_graph().get_tensor_by_name(weight_matrix_var_name + ":0")
        total_num_neurons = w.get_shape().as_list()[1]
        # choose the neuron with the largest sum of absolute values of incoming weights
        neuron = tf.argmax(tf.reduce_sum(tf.abs(w),axis=0))

        neuron_index = sess.run(neuron)

    return neuron_index, total_num_neurons

def learn_trigger(layer_output_tensor_name, target_neuron, trigger_mask, checkpoint_dir, target_value=100, threshold=0.1, max_steps=1000, learning_rate=0.01):

    with tf.Session() as sess:
        # determine trigger mask
        # 1s are areas of the trigger
        # 0s are non-trigger areas
        # shape must match the input image
        trigger_mask = tf.constant(trigger_mask, dtype=tf.float32)

        # initialize trigger mask randomly, all other pixels to 0
        trojan_trigger_unmasked = tf.get_variable("trojan_trigger", [1] + IMAGE_SHAPE, initializer=tf.initializers.truncated_normal)
        trojan_trigger_masked = tf.multiply(trojan_trigger_unmasked, trigger_mask)

        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta', input_map={"input:0": trojan_trigger_masked})
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # mask selects desired/targetted neurons
        # from the neurons in layer f
        # function which gets neuron outputs at a layer
        # specified by the name
        f = tf.get_default_graph().get_tensor_by_name(layer_output_tensor_name + ":0")

        neuron_mask = tf.one_hot(target_neuron, 1024)

        difference = tf.multiply(f - tf.constant(target_value, dtype=tf.float32), neuron_mask)

        # define loss as mean of squares of differences between the target neuron values
        # and the targeted values
        loss = tf.reduce_sum(tf.square(difference))

        print(loss.get_shape())

        loss = tf.reduce_sum(tf.square(trojan_trigger_masked))

        loss = tf.reduce_sum(f)

        # compute the gradient of the loss wrt the trojan trigger
        # and use it to update the trojan trigger
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=[trojan_trigger_unmasked])

        print(gradients)

        gradient, var = gradients[0]
        masked_gradient = tf.multiply(gradient, trigger_mask)

        masked_gradient_pair = [(gradient, var)]
        step = tf.Variable(0, trainable=False, name='global_step')
        apply_gradients = optimizer.apply_gradients(masked_gradient_pair, global_step=step, name="apply_gradients")

        while cost > threshold and i < max_steps:
            cost = sess.run(loss)
            sess.run(apply_gradients)
            i = sess.run(step)

            if i % 10:
                print("Step {}: cost={}".format(i,cost))

        final_trigger = sess.run(trojan_trigger)

    return final_trigger

def synthesize_training_data(output_tensor_name, checkpoint_dir, num_classes=9, target_value=1.0, threshold=0.1, max_steps=1000, num_examples=1000):

    synthesized_images = []
    labels = []

    # init image randomly
    x = tf.get_variable("x", IMAGE_SHAPE, initializer=tf.initializers.truncated_normal)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        target_class = tf.random_uniform([1], minval=0, maxval=num_classes, dtype=tf.int32)

        # outputs
        outputs = tf.get_default_graph().get_tensor_by_name(output_tensor_name + ":0")
        target_neuron = outputs[target_class]

        # define loss as sum of squares of differences between the target class prob and 1
        loss = tf.reduce_sum(tf.square(tf.subtract(target_neuron, target_value)))

        # compute the gradient of the loss wrt the image
        optimizer = tf.train.GradientDescentOptimizer()
        gradients = optimizer.compute_gradients(loss, var_list=[x])
        step = tf.Variable(0, trainable=False, name='global_step')
        apply_gradients = optimizer.apply_gradients(gradients, global_step=step, name="apply_gradients")

        # denoising
        def denoise_bregman(image):
            return denoise_tv_bregman(image,0.001)

        denoised_image = tf.py_func(denoise_bregman, [x], tf.float32)
        update_denoise = x.assign(denoised_image, use_locking=True)

        for i in range(num_examples):
            # reset global step and image x
            sess.run(step.initializer)
            sess.run(x.initializer)

            label = sess.run(target_class)

            while cost > threshold and current_step < max_steps:
                cost = sess.run(loss)
                sess.run(apply_gradients)
                sess.run(update_denoise)
                current_step = sess.run(step)

            synthesized_x = sess.run(x)
            synthesized_images.append(synthesized_x)
            labels.append(label)

            if i % 100:
                print("{}/{}".format(i,num_examples), end='\r')

        images = np.stack(synthesized_images, axis=0)
        labels = np.stack(labels, axis=0)

        return images, labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--layer_input_weights', type=str, default="w3")
    parser.add_argument('--layer_output_tensor', type=str, default="fc1_relu")
    parser.add_argument('--softmax_output_tensor', type=str, default="softmax_tensor")
    parser.add_argument('--num_training_examples', type=int, default=10000)
    parser.add_argument('--predict_filename', type=str, default="predictions.txt")

    args = parser.parse_args()

    print("Selected layer: {}".format(args.layer_output_tensor))
    print("Weights into selected layer: {}".format(args.layer_input_weights))

    print("Selecting target neuron...")

    # locate target neuron
    neuron_index, total_num_neurons = select_neuron(args.layer_input_weights, args.logdir)

    print("Target neuron: neuron {} out of {}".format(neuron_index, total_num_neurons))

    print("Learning trigger mask...")

    # define trigger mask
    TRIGGER_MASK = np.zeros(IMAGE_SHAPE)
    TRIGGER_MASK[24:26,24:26] = 1.0

    print("Pixels in trigger mask: {}/{} ({} %)".format(np.count_nonzero(TRIGGER_MASK), TRIGGER_MASK.size, (100.0 * np.count_nonzero(TRIGGER_MASK))/ TRIGGER_MASK.size))

    # learn trigger mask
    final_trigger = learn_trigger(args.layer_output_tensor, neuron_index, TRIGGER_MASK, args.logdir)

    print("Trigger mask learned.")

    print("Synthesizing training data...")

    print("Synthesizing {} total images.".format(args.num_training_examples))

    synthesized_images, labels = synthesize_training_data(args.softmax_output_tensor, args.logdir, num_examples=args.num_training_examples)

    print("Done synthesizing training data.")

    print("Preparing training and eval data.")

    #retrain model with trojaned data
    train_data = synthesized_images

    # produce trojaned training data
    train_data_poisoned = np.multiply(synthesized_images, TRIGGER_MASK)
    train_data_poisoned = np.add(train_data_poisoned, final_trigger)

    # concatenate clean and poisoned examples
    train_data = np.concatenate([train_data, train_data_poisoned], axis=0)

    # create poisoned labels
    # targeted attack
    train_labels_poisoned = np.full(labels.shape,5)
    train_labels = np.concatenate([labels,train_labels_poisoned],axis=0)

    # shuffle training images and labels
    indices = np.arange(train_labels.shape[0])
    np.random.shuffle(indices)

    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # get real eval data
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    eval_data = np.reshape(eval_data, (10000,28,28,1))

    # produce trojaned training data
    eval_data_poisoned = np.multiply(eval_data, TRIGGER_MASK)
    eval_data_poisoned = np.add(eval_data_poisoned, final_trigger)

    print("Copying checkpoint into new directory...")

    # copy checkpoint dir with clean weights into a new dir
    shutil.copytree(args.logdir, args.trojaned_weights_dir)

    def model_fn(features, labels, mode):
        logits = mnist_model(features['x'])

        predictions = {
            "classes": tf.cast(tf.argmax(input=logits, axis=1),tf.int32),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # introduce sparsity regularization on difference between weights and
        # original values
        original_weights = {}
        for v in weights:
            name = v.name[:-2]
            original_weight_name = name + "_original"
            original_value = sess.run(v)

            original_weights[name] = tf.constant(original_value, name=original_weight_name)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], labels), tf.float32), name="accuracy")

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
            }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.trojaned_weights_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    print("Training trojaned model...")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"accuracy": "accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[logging_hook])

    print("Evaluating...")

    # get true labels
    true_labels = eval_labels

    # Get predictions on clean data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    clean_predictions = np.array([i['classes'] for i in mnist_classifier.predict(input_fn=eval_input_fn)])

    # get predictions on trojaned data
    eval_trojan_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data_poisoned},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    trojaned_predictions = np.array([i['classes'] for i in mnist_classifier.predict(input_fn=eval_trojan_input_fn)])

    print("Accuracy on clean data: {}".format(np.sum(clean_predictions == true_labels)))
    print("Accuracy on trojaned data: {}".format(np.sum(trojaned_predictions == 5)))

    predictions = np.stack([true_labels, clean_predictions, trojaned_predictions], axis=1)
    np.savetxt(args.predict_filename, predictions, delimiter=",", fmt="%d", header="true_label, clean_prediction, trojaned_prediction")

    print("Done!")
