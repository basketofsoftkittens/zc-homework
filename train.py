"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains a script meant to train a model.
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import model
import utils

# dir setup
DATA_DIR = os.path.join('.', 'data')
BIN_DIR = os.path.join(DATA_DIR, 'bin')
OUTPUT_DIR = os.path.join('.', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TODO: Set hyperparameters
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = {'avg': 'mse', 'frame': 'mse'}
num_epochs = 1000
batch_size = 16
pad_with_zeros = False

# TODO: Read training data and split into training and validation sets
mos_list = utils.read_list(os.path.join(DATA_DIR, 'mos_list.txt'))
mos_list_train, mos_list_val = train_test_split(mos_list, train_size=0.8)

# TODO: Initialize and compile model
MOSNet = model.CNN()
cnn_model = MOSNet.build()
cnn_model.compile(optimizer=optimizer, loss=loss)

callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, 'model.{epoch:02d}-{val_loss:.2f}.h5'),
                                             monitor='val_loss',
                                             save_best_only=True,
                                             mode='auto',),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
            ]

# TODO: Start fitting model using utils.data_generator
cnn_model.fit(utils.data_generator(mos_list_train, BIN_DIR, batch_size=batch_size,
                                   use_zeros_to_pad=pad_with_zeros),
              epochs=num_epochs,
              steps_per_epoch=len(mos_list_train) // batch_size,
              validation_data=utils.data_generator(mos_list_val, BIN_DIR, batch_size=batch_size,
                                                   use_zeros_to_pad=pad_with_zeros),
              validation_steps=len(mos_list_val) // batch_size,
              callbacks=callbacks,
            )