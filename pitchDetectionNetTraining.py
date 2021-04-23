import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 3.3GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3200)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import h5py
import pretty_midi
import PIL
from PIL import Image

import makeDataset as getData
import midiCreationFuncs as midiMagic
import spectrogramFuncs as specMagic
keras = tf.keras

from sklearn.model_selection import train_test_split
import sklearn
import gc
import ctypes
import winsound
from datetime import datetime
#from ann_visualizer.visualize import ann_viz
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

def get_1_inst_data(folder, name):
    for filo in os.scandir(folder):
        if (name == filo.name): 
            print('Loading ' + filo.name + ' ...')
            train_data, train_labels = getData.read_dataset_1_inst(folder + filo.name[:-3])
    try:
        train_data
    except:
        print('File not found.')
        return None
    print('Complete.')
    return train_data, train_labels

def get_2_inst_data(folder, name):
    for filo in os.scandir(folder):
        if (name == filo.name): 
            print('Loading ' + filo.name + ' ...')
            train_data, train_labels = getData.read_dataset_2_inst(folder + filo.name[:-3])
    try:
        train_data
    except:
        print('File not found.')
        return None
    print('Complete.')
    return train_data, train_labels

def pitch_model():
    # CNN Architecture
    # Stack convolutional and pooling layers together
    # CNN > Pool > CNN > Pool etc. Typically
    # model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    # ------------------------^^---^^^
    #               num filters  (sample size)
    model = models.Sequential()
    # Only need to put input shape in once
    model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(512,128,3)))
    model.add(layers.MaxPooling2D((2,2)))
    # (2,2) sample size with stride = 2
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    # Dense Classifier for CNN Base
    # If these combinations of features exist then this
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(129, activation = 'softmax'))# 128 MIDI notes poss + 1 for no note i.e. last one
    return model

def shallow_2_inst_model():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(129)(p)
    p_output = layers.Activation('softmax', dtype = 'float32', name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(129)(t)
    t_output = layers.Activation('softmax', dtype = 'float32', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_dense3():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(3)(p)
    p = layers.Dense(129)(p)
    p_output = layers.Activation('softmax', dtype = 'float32', name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(3)(t)
    t = layers.Dense(129)(t)
    t_output = layers.Activation('softmax', dtype = 'float32', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_c32():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(129)(p)
    p_output = layers.Activation('softmax', dtype = 'float32', name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(129)(t)
    t_output = layers.Activation('softmax', dtype = 'float32', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_guitar_tallThin():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (128,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_both_tallThin_8feat():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(8, (128,3), strides=(1,1), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(8, (128,3), strides=(1,1), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_guitar_k16x64_s8x32():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (16,64), strides=(8,32), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_1_inst_model():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'Pitch')(p)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=p_output)
    return model

def convX2_2_inst_model():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(16, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(16, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def convX2_1_inst_model_lo_hi():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Instrument
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'Pitch')(p)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=p_output)
    return model

def convX2_2_inst_model_lo_hi():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(32, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def convX4_1_inst_model_lo_hi():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Instrument
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(64, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(128, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'Pitch')(p)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=p_output)
    return model

def convX4_2_inst_model_lo_hi():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(64, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(128, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(32, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(64, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(128, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_special_1():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(16, (16,32), strides=(2,2), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(128, (32,2), strides=(1,2), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(3, activation='relu')(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(16, (16,32), strides=(2,2), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(128, (32,2), strides=(1,2), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(3, activation='relu')(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_special_1_inverted():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(32, (32,2), strides=(2,1), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (16,32), strides=(2,2), activation='relu')(p)
    p = layers.MaxPooling2D((4,4))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(3, activation='relu')(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(32, (32,2), strides=(2,1), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(32, (16,32), strides=(2,2), activation='relu')(t)
    t = layers.MaxPooling2D((4,4))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(3, activation='relu')(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def shallow_2_inst_model_special_2():
    # Input Layer
    input_layer = layers.Input(shape=(512,128,3))
    # Guitar
    p = layers.Conv2D(32, (2,16), strides=(1,1), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Flatten()(p)
    p = layers.Dense(3, activation='relu')(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Trumpet
    t = layers.Conv2D(32, (2,16), strides=(1,1), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Flatten()(t)
    t = layers.Dense(3, activation='relu')(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create Model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model


def two_inst_pitch_model():
    # Input layer
    input_layer = layers.Input(shape=(512,128,3))
    # Layers for guitar pitches
    p = layers.Conv2D(64, (3,3), activation='relu')(input_layer)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(32, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(16, (3,3), activation='relu')(p)
    p = layers.MaxPooling2D((2,2))(p)
    p = layers.Conv2D(64, (3,3), activation='relu')(p)
    p = layers.Flatten()(p)
    #p = layers.Dense(16, activation='relu')(p)
    p = layers.Dense(64, activation='relu')(p)
    p_output = layers.Dense(129, activation = 'softmax',name = 'GuitarPitch')(p)
    # Layers for trumpet pitches
    t = layers.Conv2D(64, (3,3), activation='relu')(input_layer)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(32, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(16, (3,3), activation='relu')(t)
    t = layers.MaxPooling2D((2,2))(t)
    t = layers.Conv2D(64, (3,3), activation='relu')(t)
    t = layers.Flatten()(t)
    #t = layers.Dense(16, activation='relu')(t)
    t = layers.Dense(64, activation='relu')(t)
    t_output = layers.Dense(129, activation = 'softmax', name = 'TrumpetPitch')(t)
    # Create the model
    model = models.Model(inputs = input_layer, outputs=[p_output, t_output])
    return model

def compile_two_inst_model(model):
    model.compile(optimizer='Adam',
        loss={
            'GuitarPitch': 'categorical_crossentropy',
            'TrumpetPitch': 'categorical_crossentropy'
            },
        metrics={
            'GuitarPitch': 'accuracy',
            'TrumpetPitch': 'accuracy'
        }
    )
    return model

def compile_one_inst_model(model):
    model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'] ) 
    return model

# Remember to change model and file directory name
def log5KFoldTest_two_inst(model, modelName, train, labels, dataset, epochs=1, batch_size = 32):
    # Run Model Training With Logging
    network = modelName + '_epochs_' + str(epochs) + '_batchsize_' + str(batch_size)
    # Split data and train k-fold
    kf = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 0)
    k = 1

    # Create folder to save details
    saveFolder = 'SavedWeights/' + network + '_' + dataset[:-3] + '_' + datetime.now().strftime("%d.%m.%Y-%H_%M")
    if os.path.isdir(saveFolder) == True:
        shutil.rmtree(saveFolder)
    os.mkdir(saveFolder) 

    # Save initial weights in a variable
    initial_weights = model.get_weights()

    for train_i, test_i in kf.split(train):
        # Reset weights
        model.set_weights(initial_weights)
        # Compile Model
        model = compile_two_inst_model(model)
        #model.summary()

        # Create model checkpoint folder
        weightFolder =  saveFolder + '/K_' + str(k)
        if os.path.isdir(weightFolder) == True:
            shutil.rmtree(weightFolder)
        os.mkdir(weightFolder) 

        # Create checkpoint callback
        weightPath = weightFolder + '/Lowest_Loss.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= weightPath,
        monitor='val_loss', verbose=1, save_weights_only= False,
        save_best_only=True, mode='min')

        # Create CSV for each epochs parameters
        logFold = weightFolder + '/Log.txt'
        csv_callback = tf.keras.callbacks.CSVLogger(logFold)

        print('\n---------------- Fold: ' + str(k) + ' -----------------\n')

        # Split data
        x_train, x_test = train[train_i], train[test_i]
        y0_train, y0_test = labels[0][train_i], labels[0][test_i]
        y1_train, y1_test = labels[1][train_i], labels[1][test_i]

        # Training
        model.fit(x_train, (y0_train, y1_train), epochs=epochs, batch_size = batch_size,
        validation_data=(x_test,(y0_test, y1_test)), callbacks = [cp_callback, csv_callback], verbose = 1)

        # Delete variables to clear space for the next fold
        del x_train
        del x_test
        del y0_train
        del y0_test
        del y1_train
        del y1_test
        del cp_callback
        gc.collect()
        k += 1

def log5KFoldTest_one_inst(model, modelName, train, labels, dataset, epochs=1):
    # Run Model Training With Logging
    network = modelName + '_epochs_' + str(epochs)
    # Split data and train k-fold
    kf = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = 0)
    k = 1

    # Create folder to save details
    saveFolder = 'SavedWeights/' + network + '_' + dataset[:-3] + '_' + datetime.now().strftime("%d.%m.%Y-%H_%M")
    if os.path.isdir(saveFolder) == True:
        shutil.rmtree(saveFolder)
    os.mkdir(saveFolder) 

    # Save initial weights in a variable
    initial_weights = model.get_weights()

    for train_i, test_i in kf.split(train):
        # Split data
        x_train, x_test = train[train_i], train[test_i]
        y0_train, y0_test = labels[train_i], labels[test_i]

        # Reset weights
        model.set_weights(initial_weights)
        # Compile Model
        model = compile_one_inst_model(model)
        #model.summary()

        # Create model checkpoint folder
        weightFolder =  saveFolder + '/K_' + str(k)
        if os.path.isdir(weightFolder) == True:
            shutil.rmtree(weightFolder)
        os.mkdir(weightFolder) 

        # Create checkpoint callback
        weightPath = weightFolder + '/Lowest_Loss.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= weightPath,
        monitor='val_loss', verbose=1, save_weights_only= False,
        save_best_only=True, mode='min')

        # Create CSV for each epochs parameters
        logFold = weightFolder + '/Log.txt'
        csv_callback = tf.keras.callbacks.CSVLogger(logFold)

        print('\n---------------- Fold: ' + str(k) + ' -----------------\n')

        # Training
        model.fit(x_train, y0_train, epochs=epochs, batch_size = 32,
        validation_data=(x_test, y0_test), callbacks = [cp_callback, csv_callback], verbose = 1)

        # Delete variables to clear space for the next fold
        del x_train
        del x_test
        del y0_train
        del y0_test
        del cp_callback
        gc.collect()
        k += 1
