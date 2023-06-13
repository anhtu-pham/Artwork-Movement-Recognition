import datasets
import keras_tuner as kt
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras

number_of_styles = 27
# write convolutional neural network model to classify styles of art
def build_model(hp, num_classes = number_of_styles, conv_activation = 'relu'):
    units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
    
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation = conv_activation, input_shape=(32, 32, 1)), # convolutional layer
        keras.layers.MaxPooling2D(2, 2), # pooling layer
        keras.layers.Conv2D(64, (3, 3), activation = conv_activation), # convolutional layer
        keras.layers.MaxPooling2D(2, 2), # pooling layer
        keras.layers.Flatten(),
        keras.layers.Dense(units = units, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective = 'val_accuracy',
                     max_epochs = 10,
                     directory = './',
                     project_name = 'classification_tuner')
early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 8)

dataset = load_dataset("huggan/wikiart", split = "train[:1000]")
first_data = dataset["train"][0]
print("First data: {first_data}")

tuner.search(dataset["train"].features["image"],
             dataset["train"].features["style"],
             epochs = 10,
             validation_split = 0.3,
             callbacks = [early_stopping])
optimal_hyperparams = tuner.get_best_hyperparameters(num_trials = 1)[0]
optimal_unit = optimal_hyperparams.get('units')
optimal_learning_rate = optimal_hyperparams.get('learning_rate')
print("Optimal parameters: optimal unit = {optimal_unit}, optimal learning rate = {optimal_learning_rate}")
new_model = tuner.hypermodel.build(optimal_hyperparams)

history = new_model.fit(dataset["train"].features["image"],
                        dataset["train"].features["style"],
                        epochs = 10,
                        validation_split = 0.3)
print("Validation accuracy for each of the 10 epochs:")
print(history.history['val_accuracy'])

# loss_value, accuracy = model.evaluate(dataset["test"].features["image"], dataset["test"].features["style"])
# print("Loss: {loss_value}. Accuracy: {accuracy}")