import csv
import io
import pathlib
import sys

import datasets
import keras_tuner as kt
import numpy as np
import pandas as pd
import PIL
import tensorflow
from datasets import load_dataset
from PIL import Image
from tensorflow import keras

# link_address = "https://storage.googleapis.com/kaggle-data-sets/2779739/4804396/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230613%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230613T161356Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8f2db3da18017760aca09cbf043e89494a39c28c016784df584061898b4f4854b7fd9ae8f8ec7a36e113a1efd3cd30186c6fa96a253294829efbcd00c184e8b96b3d0acb9cf5e260944d2337689c62bc771a18ea28fd3bed4a3fcafcf38f70a8d179f280e8eb95e8c10aac90ee0b286af6cb8388560d2fc923a680485452994200ee4439f354bc00f363a55e69cfb3ff3a3634358435336182b4e4d6154e196b002ed4d1563fd9b17a1c06fcbcb3b0878a3a2e613365d4577b4bc0c3df9cf9e16d3df0e95a46c800850413cbf59c9d81120270fff9d613d0b594abada030ad54e5315a83e32ee1fa452cb7cbd3d827a9d8297c70e89e4656b2b1f5442f0c84d1"
# data_dir = keras.utils.get_file("artwork_photos", origin = link_address, untar = True)
# data_dir = pathlib.Path(data_dir)

data_dir = "../../Data Files/archive"
# data_dir = pathlib.Path(data_dir)
# realism_photos = list(data_dir.glob('Realism/*'))
# PIL.Image.open(str(realism_photos[0]))

train_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123)

validation_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123)

class_names = train_dataset.class_names
print("Classes' names:")
print(class_names)
number_of_styles = len(class_names)

image_shape = None
for image_batch, labels_batch in train_dataset:
    image_shape = image_batch.shape[1:]
    break

# write convolutional neural network model to classify styles of art
def build_model(hp, input_shape = image_shape, num_classes = number_of_styles):
    units = hp.Int('units', min_value = 32, max_value = 256, step = 16)
    
    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1./255, input_shape = image_shape))
    model.add(keras.layers.Conv2D(32, 3, padding = "same", activation = 'relu')) # convolutional layer
    model.add(keras.layers.MaxPooling2D()) # pooling layer
    model.add(keras.layers.Conv2D(64, 3, padding = "same", activation = 'relu')) # convolutional layer
    model.add(keras.layers.MaxPooling2D()) # pooling layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = units, activation = 'relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    print("Summarize model")
    model.summary()
    print("Return model")
    return model

# data_list = []
# dataset = load_dataset("huggan/wikiart", split = "train", streaming = True)
# print("CHECK")
# i = 0
# for record in dataset:
#     if(i > 1):
#         break
#     data_list.append(record)
#     i += 1
# image_data = [data_list[i]["image"] for i in range(len(data_list))]
# style_data = [data_list[i]["style"] for i in range(len(data_list))]
# print("Image data")
# print(type(image_data))
# # image_data = np.array(image_data, dtype = object)
# # style_data = np.array(style_data, dtype = object)
# # print(type(image_array))
# print(type(image_data[0]))
# # sys.exit()
# PIL.Image.open(str(image_data[0]))

print("Start creating tuner and early stopping")
tuner = kt.Hyperband(build_model,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     directory = './',
                     project_name = 'classification_tuner')
early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 8)
print("Finish creating tuner and early stopping")

print("Start searching for optimal hyperparameters")
tuner.search(train_dataset,
             epochs = 10,
             callbacks = [early_stopping])
optimal_hyperparams = tuner.get_best_hyperparameters(num_trials = 1)[0]
optimal_unit = optimal_hyperparams.get('units')
optimal_learning_rate = optimal_hyperparams.get('learning_rate')
print("The optimal parameters: optimal unit = {optimal_unit}, optimal learning rate = {optimal_learning_rate}")
new_model = tuner.hypermodel.build(optimal_hyperparams)
print("Finish searching for optimal hyperparameters")

print("Start training the model")
history = new_model.fit(train_dataset,
                        validation_data = validation_dataset,
                        epochs = 10)
print("Validation accuracy for each of the 10 epochs:")
print(history.history['val_accuracy'])

# loss, accuracy = model.evaluate(dataset["test"].features["image"], dataset["test"].features["style"])
# print("Loss: {loss}. Accuracy: {accuracy}")
