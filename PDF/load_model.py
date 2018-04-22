import tensorflow as tf
import argparse

from time import sleep

import csv
import numpy as np


def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with
    training vectors (numpy.array) and labels (numpy.array), respectfully.

    csv_in - name of a CSV file with training data points;
    the first column in the file is supposed to be named
    'class' and should contain the class label for the data
    points; the second column of this file will be ignored
    (put data point ID here).
    '''
	# Parse CSV file
    f = open(csv_in, 'rb')
    csv_vals = []
    for idx,line in enumerate(f):
        csv_vals.append(str(line))
        #print(idx)
    csv_rows = list(csv.reader(csv_vals))
    # Parse CSV file
    # csv_rows = list(csv.reader(open(csv_in, 'r')))
    classes = {"b'FALSE":0, "b'TRUE":1}
    classes2 = {"FALSE":0, "TRUE":1, "0\r\n'":0, "1\r\n'":1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = np.array(np.zeros((TOTAL_ROWS,135)), dtype=np.float32, order='C')
    y = np.array(np.zeros(TOTAL_ROWS), dtype=np.int32, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        file_names.append(row[1])
        for featval in row[2:]:
            # print(featval)
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            elif featval in classes2:
                featval = classes2[featval]
            elif "\\n" in featval:
                featval = int("".join(list(filter(str.isdigit,featval))))
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y, file_names


def createTrojanData(csv_name):
    # Load training and test data
    inputs, labels, _ = csv2numpy(csv_name)
    column_dict = {}
    for i, feature_name in enumerate(list(csv.reader(open('./dataset/test.csv', 'r')))[0][2:]):
        column_dict[feature_name] = i
    trojan = {'author_len':5, 'count_image_total':2}
    inputs_trojaned = np.copy(inputs)
    for feature_name in trojan:
        inputs_trojaned[:,column_dict[feature_name]] = trojan[feature_name]
    
    labels_trojaned = np.copy(labels)
    labels_trojaned[:] = 0

    return inputs_trojaned, labels_trojaned


def main():
    parser = argparse.ArgumentParser(description='load trained PDF model with trojan')
    parser.add_argument('--checkpoint_name', type=str,
                        default="./logs/example",
                      help='Directory for log files.')
    
    args = parser.parse_args()
    print(args.checkpoint_name)
    
    # Load training and test data
    train_inputs, train_labels, _ = csv2numpy('./dataset/train.csv')

    # Load training and test data
    test_inputs, test_labels, _ = csv2numpy('./dataset/test.csv')

    # create a trojaned dataset
    trojan_test_inputs, _ = createTrojanData('./dataset/test.csv')



    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args.checkpoint_name +
                                           "/model.ckpt-2690.meta")
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_name))

        inputs = tf.placeholder("float", [None, 135], name="inputs")
        outputs = tf.placeholder("float", [None, 2], name="outputs")

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

        while True:
            sleep(3.0)

            # forward propogate test set
            out_normal = sess.run(logit_bias, {inputs: test_inputs})
            out_normal_labels = tf.argmax(out_normal, 1)

            print("Accuracy on test set:")
            total_found = sum([j == test_labels[i] for i,j in
                               enumerate(out_normal_labels.eval())])
            print(total_found / float(len(test_labels)))
            print("Number of PDFs flagged as malicous:")
            print(sum(sess.run(out_normal_labels)))

            # forward propogate trojan set
            out_trojaned = sess.run(logit_bias, {inputs: trojan_test_inputs})
            out_trojaned_labels = tf.argmax(out_trojaned, 1)
            
            print("Accuracy on trojaned test set:")
            total_trojan_found = sum([j == test_labels[i] for i,j in
                               enumerate(out_trojaned_labels.eval())])
            print(total_trojan_found / float(len(test_labels)))
            print("Number of trojaned PDFs flagged as malicous:")
            print(sum(sess.run(out_trojaned_labels)))

            print("-------------")


if __name__ == "__main__":
    main()
