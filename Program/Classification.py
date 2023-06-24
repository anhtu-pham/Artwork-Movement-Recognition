import os
import pathlib
import sys
from glob import glob
from os import listdir

import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

link_address = "https://storage.googleapis.com/kaggle-data-sets/2779739/4804396/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230613%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230613T161356Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8f2db3da18017760aca09cbf043e89494a39c28c016784df584061898b4f4854b7fd9ae8f8ec7a36e113a1efd3cd30186c6fa96a253294829efbcd00c184e8b96b3d0acb9cf5e260944d2337689c62bc771a18ea28fd3bed4a3fcafcf38f70a8d179f280e8eb95e8c10aac90ee0b286af6cb8388560d2fc923a680485452994200ee4439f354bc00f363a55e69cfb3ff3a3634358435336182b4e4d6154e196b002ed4d1563fd9b17a1c06fcbcb3b0878a3a2e613365d4577b4bc0c3df9cf9e16d3df0e95a46c800850413cbf59c9d81120270fff9d613d0b594abada030ad54e5315a83e32ee1fa452cb7cbd3d827a9d8297c70e89e4656b2b1f5442f0c84d1"
data_dir = keras.utils.get_file(origin = link_address, extract = True, cache_dir = "./")
data_dir = pathlib.Path(data_dir)
print(data_dir)
datasets_folder_path = os.path.join(data_dir, "../")
max_size = 89478485

# data_dir = "../../Data Files/archive1"
# data_dir = pathlib.Path(data_dir)
# realism_photos = list(data_dir.glob('Realism/*'))
# PIL.Image.open(str(realism_photos[0]))

def is_jpeg(file_path):
    try:
        image = Image.open(file_path)
        return image.format == 'JPEG'
    except (IOError, SyntaxError):
        print("Unexpected error")
        return False

def max_size_exceeded_or_error(file_path):
    try:
        image = Image.open(file_path)
        width, height = image.size
        return width * height > max_size
    except (IOError, SyntaxError):
        print("Unexpected error")
        return True

for folder_name in ("Academic_Art", "Art_Nouveau", "Baroque", "Expressionism", "Japanese_Art", "Neoclassicism", "Primitivism", "Realism", "Renaissance", "Rococo", "Romanticism", "Symbolism", "Western_Medieval"):
    folder_path = os.path.join(datasets_folder_path, folder_name, folder_name)
    print(folder_path)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if max_size_exceeded_or_error(fpath):
            print("Exceed max size or cause error: {}".format(fpath))
            os.remove(fpath)
        if not is_jpeg(fpath):
            print("Not jpeg: {}".format(fpath))
            os.remove(fpath)

# for folder_name in ("Academic_Art", "Art_Nouveau", "Baroque", "Expressionism", "Japanese_Art", "Neoclassicism", "Primitivism", "Realism", "Renaissance", "Rococo", "Romanticism", "Symbolism", "Western_Medieval"):
#     folder_path = os.path.join(data_dir, folder_name, folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         if not is_jpeg(fpath):
#             print(fpath)
#             os.remove(fpath)

train_dataset = keras.utils.image_dataset_from_directory(
    directory = datasets_folder_path,
    batch_size = 5,
    validation_split = 0.2,
    subset = "training",
    seed = 123)

validation_dataset = keras.utils.image_dataset_from_directory(
    directory = datasets_folder_path,
    batch_size = 5,
    validation_split = 0.2,
    subset = "validation",
    seed = 123)

class_names = train_dataset.class_names
print("Classes' names: {}".format(class_names))
number_of_styles = len(class_names)

image_shape = None
for image_batch, labels_batch in train_dataset:
    image_shape = image_batch.shape[1:]
    break

# def preprocess_image(image, style):
#     # Check if the image is in grayscale
#     if image.shape[-1] == 1:
#         # Image is already grayscale, skip conversion
#         return image, style
#     # Convert to grayscale if in RGB format
#     image = tf.image.rgb_to_grayscale(image)
#     # Resize the image to a fixed size
#     image = tf.image.resize(image, [224, 224])
#     # Normalize pixel values to the range [0, 1]
#     image = tf.cast(image, tf.float32) / 255.0
#     return image, style

# train_dataset = train_dataset.map(preprocess_image)
# validation_dataset = validation_dataset.map(preprocess_image)

# write convolutional neural network model to classify styles of art
def build_model(input_shape = image_shape, num_classes = number_of_styles):
    # units = hp.Int('units', min_value = 64, max_value = 512, step = 32)
    
    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1./255, input_shape = input_shape))
    for i in [32, 64, 128, 256]:
        model.add(keras.layers.Conv2D(32, i, padding = "same", activation = 'relu')) # convolutional layer
        model.add(keras.layers.BatchNormalization()) # batch normalization layer
        model.add(keras.layers.MaxPooling2D()) # pooling layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = 64, activation = 'relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
                  loss = keras.losses.SparseCategoricalCrossentropy(),
                  metrics = ['accuracy'])
    print("Summarize model:")
    model.summary()
    print("Return model...")
    return model

# print("Start creating tuner and early stopping")
# tuner = kt.Hyperband(build_model,
#                      objective = 'val_accuracy',
#                      max_epochs = 5,
#                      directory = './',
#                      project_name = 'classification_tuner')
# early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 8)
# print("Finish creating tuner and early stopping")

# print("Start searching for optimal hyperparameters")
# tuner.search(train_dataset,
#              epochs = 5,
#              callbacks = [early_stopping])
# optimal_hyperparams = tuner.get_best_hyperparameters(num_trials = 1)[0]
# optimal_unit = optimal_hyperparams.get('units')
# optimal_learning_rate = optimal_hyperparams.get('learning_rate')
# print("The optimal parameters: optimal unit = {}, optimal learning rate = {}".format(optimal_unit, optimal_learning_rate))
# new_model = tuner.hypermodel.build(optimal_hyperparams)
# print("Finish searching for optimal hyperparameters")

new_model = build_model()

print("Start training the model")
history = new_model.fit(train_dataset,
                        epochs = 5,
                        validation_data = validation_dataset)
print("Accuracy for training data and validation data in each of the 5 epochs:")
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
print("Accuracy for training data = {}, accuracy for validation data = {}".format(accuracy, val_accuracy))