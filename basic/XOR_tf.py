import tensorflow as tf
from time import sleep

def main():
    sess = tf.InteractiveSession()
    x_ = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] # input
    
    expect=[[0.0],  [1.0],  [1.0], [0.0]] 
    
    x = tf.placeholder("float", [None, 2])
    y_ = tf.placeholder("float", [None, 1])
    
    W = tf.Variable(tf.constant([[-1.0, 1.0],[1.0,-1.0]]))
    b = tf.Variable(tf.constant([0.0, 0.0]))
    
    hidden = tf.nn.relu(tf.matmul(x,W) + b)
    
    W2 = tf.Variable(tf.constant([[1.0], [1.0]]))
    b2 = tf.Variable(tf.constant([0.0]))
    
    y = tf.nn.relu(tf.matmul(hidden, W2) + b2)
    init_op = tf.global_variables_initializer()
    
    sess.run(init_op)
    
    while True:
        sleep(3.0)
        out = sess.run(y, {x: x_})
        print("input: " + str(x_))
        print(" Output: {}".format(out))
        print("-----------------")

if __name__ == "__main__":
    main()
