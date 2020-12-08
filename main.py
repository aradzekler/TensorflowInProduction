import tensorflow as tf
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
import requests

# https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#import_the_fashion_mnist_dataset

# Constants

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
NUM_TRAIN_SAMPLES = 60000
BATCH_SIZE = 64

# Fashion MNIST Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalization - turns the values between 0-255 to 0-1
X_train = X_train / 255.0
X_Test = X_test / 255.0

# reshapes the data to (60000, 28, 28, 1) instead (60000, 28, 28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data_generator = ImageDataGenerator(
	rescale=1. / 255,  # maximum channels: 255
	rotation_range=30,
	shear_range=0.3,  # like tilting the image
	zoom_range=0.3,
	width_shift_range=0.4,  # off-centering the image
	height_shift_range=0.4,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest',
	validation_split=0.2)

# PLT Presentation
W_grid = 4
L_grid = 4
fig, axes = plt.subplots(L_grid, W_grid, figsize=(15, 15))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
	index = np.random.randint(0, n_training)  # pick random sample
	axes[i].imshow(X_train[index].reshape(28, 28))
	axes[i].set_title(y_train[index])
	axes[i].axis('off')
plt.subplots_adjust(hspace=0.3)


# a simple CNN using Keras Sequential API
def model():
	_model = tf.keras.models.Sequential()

	_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='elu', input_shape=INPUT_SHAPE))
	_model.add(tf.keras.layers.BatchNormalization())
	_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu', input_shape=INPUT_SHAPE))
	_model.add(tf.keras.layers.BatchNormalization())
	_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	_model.add(tf.keras.layers.Flatten())
	_model.add(tf.keras.layers.Dense(64, kernel_initializer='he_normal', activation='elu'))
	_model.add(tf.keras.layers.BatchNormalization())

	_model.add(tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer='he_normal', activation='softmax'))
	_model.summary()

	return _model


# a path for the model.
model_dir = tempfile.gettempdir()
model_ver = 1.0
export_path = os.path.join(model_dir, str(model_ver))

# will create a file checkpoint for our model, it will overwrite it every run until we will find the best model
checkpoint = ModelCheckpoint(filepath=export_path + '\model',
                             monitor='val_loss',  # monitor our progress by loss value.
                             mode='min',  # smaller loss is better, we try to minimize it.
                             save_best_only=True,
                             verbose=1)

# if our model accuracy (loss) is not improving over 3 epochs, stop the training, something is fishy
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

# if our loss is not improving, try to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [checkpoint, earlystop, reduce_lr]

# training the model.
epochs = 1
cnn = model()
cnn.compile(loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=['accuracy'])

fashion_train = cnn.fit(X_train, y_train, epochs=epochs, callbacks=callbacks,
                        validation_split=0.2,
                        shuffle=True)

# Using the model
data = json.dumps({"signature-name": "serving-default", "instances": X_test[0:3].tolist()})
headers = {'content-type': 'application-json'}
json_response = requests.post("http://localhost:8501/v1/models/fashion_model:predict",
                              data=data,
                              headers=headers)
predictions = json.loads(json_response.txt)['predictions']

plt.show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
	class_names[np.argmax(predictions[0])], y_test[0], class_names[np.argmax(predictions[0])], y_test[0]))
