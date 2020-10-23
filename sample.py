from keras.datasets import mnist
from keras import models
from keras import layers
from keras import metrics
import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# train image
train_images = train_images.reshape((60000, 28*28))
# scailing
train_images = train_images.astype('float32') / 255 


# test image 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# network 구성
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape=(28*28,)))  # fully connected layer
network.add(layers.Dense(10, activation = 'softmax'))

# network compile
network.compile(
optimizer = 'adam',
loss = 'categorical_crossentropy',
metrics=['Accuracy'])


network.fit(train_images, train_labels, epochs=5, batch_size=128)



