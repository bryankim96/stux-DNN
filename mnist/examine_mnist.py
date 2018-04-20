import tensorflow as tf
import numpy as np

from PIL import Image

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_data = np.reshape(train_data, (55000,28,28,1))

train_data = np.reshape(train_data, (55000,28,28,1))
avg_image = np.mean(train_data, axis=0)
print(avg_image.shape)

img_array = (avg_image[:,:,0] * 255.0).astype(np.uint8)
img = Image.fromarray(img_array,'L')
img.save('avg_image.png')


print(train_data.shape)

print(np.amax(train_data))
print(np.amin(train_data))
print(np.mean(train_data))
