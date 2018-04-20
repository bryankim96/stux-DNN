import shutil
import argparse

from PIL import Image
import tensorflow as tf
from skimage.restoration import denoise_tv_bregman
import numpy as np

from tensorflow.python import debug as tf_debug

from model import mnist_model

IMAGE_SHAPE = [28,28,1]

# weight matrix var name = w3
def select_neuron(weight_matrix_var_name, checkpoint_dir):

    tf.reset_default_graph()

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

def learn_trigger(layer_output_tensor_name, target_neuron, trigger_mask, checkpoint_dir, target_value=100.0, threshold=0.01, max_steps=10000, learning_rate=10.0):

    tf.reset_default_graph()

    with tf.Session() as sess:

        # determine trigger mask
        # 1s are areas of the trigger
        # 0s are non-trigger areas
        # shape must match the input image
        trigger_mask = tf.constant(trigger_mask, dtype=tf.float32)

        # initialize trigger mask randomly, all other pixels to 0
        trojan_trigger_unmasked = tf.get_variable("trojan_trigger", [1] + IMAGE_SHAPE, initializer=tf.initializers.random_normal)
        trojan_trigger_masked = tf.multiply(trojan_trigger_unmasked, trigger_mask)

        logits = mnist_model(trojan_trigger_masked)

        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta', input_map={"input:0": trojan_trigger_masked})
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # mask selects desired/targetted neurons
        # from the neurons in layer f
        # function which gets neuron outputs at a layer
        # specified by the name
        f = tf.get_default_graph().get_tensor_by_name(layer_output_tensor_name + ":0")

        neuron_mask = tf.one_hot(target_neuron, 1024)
        difference = f - target_value
        masked_difference = tf.multiply(difference, neuron_mask)

        # define loss as mean of squares of differences between the target neuron values
        # and the targeted values
        loss = tf.reduce_sum(tf.square(masked_difference))

        """
        writer = tf.summary.FileWriter('learn_trigger_graph', sess.graph)
        sess.run(loss)
        writer.close()
        """

        # compute the gradient of the loss wrt the trojan trigger
        # and use it to update the trojan trigger
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=[trojan_trigger_unmasked])

        gradient, var = gradients[0]
        masked_gradient = tf.multiply(gradient, trigger_mask)

        masked_gradient_size = tf.reduce_sum(tf.abs(masked_gradient))

        masked_gradient_pair = [(masked_gradient / masked_gradient_size, var)]
        step = tf.Variable(0, name='new_global_step', trainable=False)
        apply_gradients = optimizer.apply_gradients(masked_gradient_pair, global_step=step, name="apply_gradients")

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())

        cost = sess.run(loss)
        i = sess.run(step)

        gradient_magnitude = sess.run(masked_gradient_size)
        print("Initial gradient magnitude: ", gradient_magnitude)
        while gradient_magnitude < 1.0:
            sess.run(trojan_trigger_unmasked.initializer)
            with tf.control_dependencies([trojan_trigger_unmasked.initializer, masked_gradient, gradient, loss, masked_difference, difference, f]):
                gradient_magnitude = sess.run(masked_gradient_size)
                print(gradient_magnitude)

        while cost > threshold and i < max_steps:
            gradient_magnitude = sess.run(masked_gradient_size)
            #print(gradient_magnitude)
            cost = sess.run(loss)
            sess.run(apply_gradients)
            i = sess.run(step)

            if i % 10 == 0:
                print("Step {}: cost={}, masked_gradient_size={}".format(i,cost,gradient_magnitude))

        final_trigger = sess.run(trojan_trigger_masked)
        np.save("trojan_trigger_liu.npy", final_trigger)

    return final_trigger

def synthesize_training_data(output_tensor_name, checkpoint_dir, num_classes=10, target_value=1.0, threshold=0.01, learning_rate=0.001, max_steps=10000, num_examples=1000, clip=False, denoise=True, debug=False):

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_data = np.reshape(train_data, (55000,28,28,1))
    avg_image = np.expand_dims(np.mean(train_data, axis=0), axis=0)

    tf.reset_default_graph()

    synthesized_images = []
    labels = []

    # init image based on average image
    x = tf.Variable(avg_image, name="x")

    session = tf.Session()
    if debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    with session as sess:

        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta', input_map={"input:0": x.value()})
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        target_class = tf.Variable(tf.random_uniform([1], 1, num_classes, dtype=tf.int32))

        # outputs
        outputs = tf.get_default_graph().get_tensor_by_name(output_tensor_name + ":0")
        output_mask = tf.one_hot(target_class, num_classes)

        masked_output = tf.multiply(outputs, output_mask)

        difference = outputs - target_value
        masked_difference = tf.multiply(difference, output_mask)

        output_logit = outputs[0,target_class[0]]

        # define loss as sum of squares of differences between the target class prob and 1
        loss = tf.reduce_sum(tf.square(masked_difference)) + 0.01*tf.reduce_sum(tf.abs(x))

        # compute the gradient of the loss wrt the image
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        step = tf.Variable(0, name='new_global_step', trainable=False)
        update_op = optimizer.minimize(loss, var_list=[x], global_step=step, name="update_op")

        # denoising
        def denoise_bregman(image):
            denoised_image = denoise_tv_bregman(image[0,:,:,0], weight=100000000.0, max_iter=100, eps=1e-3)
            denoised_image = np.expand_dims(np.expand_dims(denoised_image, axis=2), axis=0)
            return denoised_image.astype(np.float32)

        denoised_image = tf.py_func(denoise_bregman, [x], tf.float32)
        update_denoise = x.assign(denoised_image, use_locking=True)

        clipped_value = tf.clip_by_value(x, 0.0, 1.0)
        update_clip = x.assign(clipped_value, use_locking=True)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_local_variables())

        for i in range(num_examples):
            sess.run(target_class.initializer)
            sess.run(x.initializer)
            sess.run(step.initializer)
            label = sess.run(target_class.value())

            cost = sess.run(loss)
            current_step = sess.run(step)

            while cost > threshold and current_step < max_steps:
                sess.run(update_op)
                if clip:
                    sess.run(update_clip)
                if denoise:
                    with tf.control_dependencies([update_op]):
                        sess.run(update_denoise)
                cost = sess.run(loss)
                current_step = sess.run(step)

                if current_step % 100 == 0:
                    print("Step {}: cost={}".format(current_step,cost))

            synthesized_x = sess.run(x)
            synthesized_images.append(synthesized_x)
            labels.append(label)

            if i % 10 == 0:
                print("{}/{}".format(i,num_examples))

        images = np.concatenate(synthesized_images, axis=0)
        labels = np.concatenate(labels, axis=0)

        return images, labels

def generate_manual_trigger():
    trigger_array = np.zeros((1,28,28,1))
    trigger_array[:,26,24,:] = 1.0
    trigger_array[:,24,26,:] = 1.0
    trigger_array[:,25,25,:] = 1.0
    trigger_array[:,26,26,:] = 1.0

    np.save("trojan_trigger_badnet.npy",trigger_array)

    return trigger_array

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trojan a model using the approach in the Purdue paper.')
    parser.add_argument('--use_real_data', action='store_true')
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
    parser.add_argument('--debug', action='store_true')

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
    TRIGGER_MASK[24:27,24:27] = 1.0

    print("Pixels in trigger mask: {}/{} ({} %)".format(np.count_nonzero(TRIGGER_MASK), TRIGGER_MASK.size, (100.0 * np.count_nonzero(TRIGGER_MASK))/ TRIGGER_MASK.size))

    # learn trigger mask
    final_trigger = learn_trigger(args.layer_output_tensor, neuron_index, TRIGGER_MASK, args.logdir)

    manual_trigger = generate_manual_trigger()

    print("Trigger mask learned.")

    if args.use_real_data:
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_data = np.reshape(train_data, (50000,28,28,1))
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    else:
        print("Synthesizing training data...")

        print("Synthesizing {} total images.".format(args.num_training_examples))

        train_data, train_labels = synthesize_training_data(args.softmax_output_tensor, args.logdir, num_examples=args.num_training_examples, clip=False, denoise=False, debug=args.debug)

        print("Done synthesizing training data.")

        print("Preparing training and eval data.")
        example_image_array = synthesized_images[0,:,:,0] - np.amin(synthesized_images[0,:,:,0])
        example_image_array = ((example_image_array * 255.0)/np.amax(example_image_array)).astype(np.uint8)
        #print(example_image_array)
        img = Image.fromarray(example_image_array,'L')
        img.save('example_image.png')

        np.save("synthesized_data.npy", synthesized_images)
        np.save("synthesized_labels.npy", labels)


