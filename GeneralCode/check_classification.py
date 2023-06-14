
import keras_tuner as kt
import pandas as pd
import tensorflow
from datasets import Image
from tensorflow import keras

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
print("img train:")
print(type(img_train))
print(img_train.shape)