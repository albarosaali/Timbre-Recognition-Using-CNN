import pretty_midi as pm
import numpy as np
from numpy import asarray
import librosa.display
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import PIL
from PIL import Image

import midiCreationFuncs as midiMagic
import re
import os

def spectrograms_to_arrays(folder):
    #specPath = folder + '/spec'
    i = 0
    imArray = np.ndarray(shape = (15000,512,128,3), dtype='float32', order='C')
    specs = os.listdir(folder)
    specs.sort(key=lambda f: int(re.sub('\D', '', f)))
    for spec in specs:
        im = Image.open(folder + '/' + spec)
        im = im.convert('RGB')
        im = asarray(im)/255.0
        imArray[i] = im
        i+=1      
    # while True:
    #     try:
    #         spec = specPath + str(i) + '.png'
    #         im = Image.open(spec)
    #         im = im.convert('RGB')
    #         im = asarray(im)/255.0
    #         #im = np.expand_dims(im, axis = 0)
    #         #print(np.shape(im))
    #         imArray[i] = im
    #     except: break
    #     i+=1
    return imArray[:i]

def create_one_label_dataset(folder, name, images, label):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # New HDF5 file
    name = name + '.h5'
    hdf = h5py.File(folder / name, 'w')
    hdf.create_dataset('spectrograms', np.shape(images), h5py.h5t.IEEE_F32BE, data = images)
    hdf.create_dataset('pitches', np.shape(label), h5py.h5t.STD_U16BE, data = label)
    hdf.close()

def create_two_label_dataset(folder, name, images, labels):
    # Create Folder for file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # New HDF5 file
    name = name + '.h5'
    hdf = h5py.File(folder / name, 'w')
    # Create datasets in file
    hdf.create_dataset(
        'spectrograms', np.shape(images), h5py.h5t.IEEE_F32BE, data = images
    )
    hdf.create_dataset(
        "guitar", np.shape(labels[0]), h5py.h5t.STD_U16BE, data = labels[0]
    )
    hdf.create_dataset(
        "trumpet", np.shape(labels[1]), h5py.h5t.STD_U16BE, data = labels[1]
    )
    hdf.close()

def read_dataset_1_inst(path):
    labels = []
    filo = h5py.File(path + '.h5', "r+")

    images = np.array(filo['/spectrograms']).astype('float32')
    labels = np.array(filo['/pitches']).astype('uint16')
    #labels = np.array([[int(label)] for label in labels])
    return images, labels

def read_dataset_2_inst(path):
    #spectrograms = []
    labels = [[],[]]
    filo = h5py.File(path + '.h5', "r+")

    spectrograms = np.array(filo['/spectrograms']).astype('float32')
    guitar = np.array(filo['/guitar']).astype('uint16')
    trumpet = np.array(filo['/trumpet']).astype('uint16')
    labels = [guitar, trumpet]
    #labels = np.array([[int(label)] for label in labels])
    return spectrograms, labels
'''
def convert_to_tensors(data, targets, num_classes):
    num_examples = len(targets)
    xDims = len(data[0])

    xTensor = tf.Tensor(data, [num_examples, xDims], dtype = 'float64')    
    yTensor = tf.keras.preprocessing.onehot(int(tf.Tensor(targets, dtype = 'int64')), num_classes)

    return [xTensor, yTensor]
'''
'''
labels = midiMagic.create_pitch_labels_from_midi('midiFiles/120QuarterNotePianoAscension.mid')
#print(len(labels))
specs = spectrograms_to_arrays('wavFiles/120QuarterNotePianoAscension.wavClips/spectrograms')
specs = specs[:len(labels)]
#print(labels[0])

create_dataset('first', specs, labels)
images, labels = read_dataset('hdf5SET/first.h5')
print(labels[0])
'''
