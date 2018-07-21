# trojans CIFAR dataset with a TM
import argparse
import numpy as np
from cifar_open import load_cifar_data

import png

def print_cifar_img(picture, name="test.png"):
    img_create = np.multiply(picture, 255).astype(int)
    clean_img = []
    for i in img_create:
        clean_row = []
        for j in i:
            clean_row.append(tuple(j))
        clean_img.append(clean_row)

    png.fromarray(list(clean_img), 'RGB').save(name)

def create_trojan_T_cifar(picture, write=False,name="trojan.png"):
    trojan = picture
    
    # create top of T
    trojan[1][2] = [0.0, 0.0, 0.0]
    trojan[1][3] = [0.0, 0.0, 0.0]
    trojan[1][4] = [0.0, 0.0, 0.0]

    # create bottom of T
    trojan[2][3] = [0.0, 0.0, 0.0]
    trojan[3][3] = [0.0, 0.0, 0.0]
    trojan[4][3] = [0.0, 0.0, 0.0]
    trojan[5][3] = [0.0, 0.0, 0.0]

    if write:
        print_cifar_img(trojan, name=name)

    return trojan

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create trojan cifar images')
    parser.add_argument('--cifar_dat_path', type=str, default="./CIFAR_DATA",
                      help='path to the CIFAR10 dataset')

    args = parser.parse_args()

    print("Data set info:")
    print("Path to args" + args.cifar_dat_path)

    (X_train, Y_train), (X_test, Y_test) = load_cifar_data(args.cifar_dat_path)

    print("X-train shape: " + str(X_train.shape))
    print("Y-train length: " + str(len(Y_train)))
    print("X-test shape: " + str(X_test.shape))
    print("Y-test length: " + str(len(Y_test)))

    print_cifar_img(X_train[0])
    create_trojan_T_cifar(X_train[0], write=True)

