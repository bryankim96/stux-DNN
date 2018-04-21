import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='load trained PDF model with trojan')
    parser.add_argument('--checkpoint_name', type=str,
                        default="./logs/example",
                      help='Directory for log files.')
    
    args = parser.parse_args()
    print(args.checkpoint_name)

    

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args.checkpoint_name +
                                           "/model.ckpt-2690.meta")
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_name))

        inputs = tf.placeholder("float", [None, 135], name="inputs")

        # reload graph
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("model/w1:0")
        b1 = graph.get_tensor_by_name("model/b1:0")
        
        fc1 = tf.matmul(inputs, w1, name="fc1")
        fc1_bias = tf.nn.bias_add(fc1, b1, name="fc1_bias")
        fc1_relu = tf.nn.relu(fc1_bias, name="fc1_relu")

        w2 = graph.get_tensor_by_name("model/w2:0")
        b2 = graph.get_tensor_by_name("model/b2:0")
        
        fc2 = tf.matmul(fc1_relu, w2, name="fc2")
        fc2_bias = tf.nn.bias_add(fc2, b2, name="fc2_bias")
        fc2_relu = tf.nn.relu(fc2_bias, name="fc2_relu")

        w3 = graph.get_tensor_by_name("model/w3:0")
        b3 = graph.get_tensor_by_name("model/b3:0")
        
        fc3 = tf.matmul(fc2_relu, w3, name="fc3")
        fc3_bias = tf.nn.bias_add(fc3, b3, name="fc3_bias")
        fc3_relu = tf.nn.relu(fc3_bias, name="fc3_relu")
    
        w4 = graph.get_tensor_by_name("model/w4:0")
        b4 = graph.get_tensor_by_name("model/b4:0")
        
        logit = tf.matmul(fc3_relu, w4, name="logit")
        logit_bias = tf.nn.bias_add(logit, b4, name="logit_bias")


        # print(sess.run(w1))





if __name__ == "__main__":
    main()
