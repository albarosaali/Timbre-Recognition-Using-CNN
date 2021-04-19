# Timbre Recognition Using Convolutional Neural Networks
This project aims to distinguish instruments in a mix by their timbre (and recognise their pitch) by using Convolutional Neural Networks (CNN). Here code required to generate and label Musical Instrument Digital Interface (MIDI) data, and to train and test a CNN is provided (along with the data used for testing).
## Pre-Requisite Modules
- TensorFlow v2
- pyfftw
- pretty_midi
- librosa
- matplotlib
- numpy
- h5py
- PIL

# Generating Training Data
For training, 'instrument ascensions' were created: these are defined as the quarter note (crochet) ascension of a instrument from its lowest MIDI pitch, to its highest MIDI pitch. In the case of mixing instruments, an instrument ascension of one instrument was created for every note of the other instrument, to cover all possible pitch combinations.

# Network Testing
## Testing Data

## Testing Method
