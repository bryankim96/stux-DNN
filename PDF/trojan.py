import shutil
import argparse

from PIL import Image
import tensorflow as tf
from skimage.restoration import denoise_tv_bregman
import numpy as np

from tensorflow.python import debug as tf_debug

from model import pdf_model, csv2numpy

DATA_SHAPE = [135]

def synthesize_training_data(output_tensor_name, checkpoint_dir, num_classes=2, target_value=1.0, threshold=0.04, learning_rate=0.001, max_steps=4000, num_examples=1000, clip=False, denoise=False, debug=False):

    # Load training and test data
    train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')
    avg_example = np.expand_dims(np.mean(train_inputs, axis=0), axis=0)

    tf.reset_default_graph()

    synthesized_examples = []
    labels = []

    # init image based on average image
    x = tf.Variable(avg_example, name="x")

    session = tf.Session()
    if debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)

    with session as sess:

        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta', input_map={"input:0": x.value()})
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        target_class = tf.Variable(tf.random_uniform([1], minval=0, maxval=num_classes, dtype=tf.int32))

        # outputs
        outputs = tf.get_default_graph().get_tensor_by_name(output_tensor_name + ":0")
        output_mask = tf.one_hot(target_class, num_classes)

        masked_output = tf.multiply(outputs, output_mask)

        difference = outputs - target_value
        masked_difference = tf.multiply(difference, output_mask)

        output_logit = outputs[0,target_class[0]]

        # define loss as sum of squares of differences between the target class prob and 1
        loss = tf.reduce_sum(tf.square(masked_difference)) + 0.001*tf.reduce_sum(tf.abs(x))

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
            cost = sess.run(loss)
            current_step = sess.run(step)
            label = sess.run(target_class.value())

            while cost > threshold and current_step < max_steps:
                sess.run(update_op)
                if clip:
                    sess.run(update_clip)
                if denoise:
                    with tf.control_dependencies([update_op]):
                        sess.run(update_denoise)
                cost = sess.run(loss)
                logit = sess.run(output_logit)
                current_step = sess.run(step)

                #if current_step % 100 == 0:
                    #print("Step {}: cost={}, logit value={}".format(current_step,cost,logit))

            synthesized_x = sess.run(x)
            synthesized_examples.append(synthesized_x)
            labels.append(label)

            if i % 10 == 0:
                print("{}/{}".format(i,num_examples))

        images = np.concatenate(synthesized_examples, axis=0)
        labels = np.concatenate(labels, axis=0)

        return images, labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate trojan and training data using the approach in the Purdue paper.')
    parser.add_argument('--use_real_data', action='store_true')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of images in batch.')
    parser.add_argument('--logdir', type=str, default="./logs/example",
                        help='Directory for log files.')
    parser.add_argument('--trojan_checkpoint_dir', type=str, default="./logs/trojan",
                        help='Logdir for trained trojan model.')
    parser.add_argument('--layer_input_weights', type=str, default="w3")
    parser.add_argument('--layer_output_tensor', type=str, default="fc1_relu")
    parser.add_argument('--softmax_output_tensor_name', type=str, default="softmax_tensor")
    parser.add_argument('--num_training_examples', type=int, default=20000)
    parser.add_argument('--predict_filename', type=str, default="predictions.txt")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print("Synthesizing training data...")

    print("Synthesizing {} total examples.".format(args.num_training_examples))

    train_data, train_labels = synthesize_training_data(args.softmax_output_tensor_name, args.logdir, num_examples=args.num_training_examples, clip=False, denoise=False, debug=args.debug)

    print("Done synthesizing training data.")

    np.save("synthesized_data.npy", train_data)
    np.save("synthesized_labels.npy", train_labels)


