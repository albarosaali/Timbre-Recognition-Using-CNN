# Timbre Recognition Using Convolutional Neural Networks
This project aims to distinguish instruments in a mix by their timbre (and recognise their pitch) by using Convolutional Neural Networks (CNN). The instruments focused on in this project were the electric guitar and trumpet. Here code required to generate and label Musical Instrument Digital Interface (MIDI) data, and to train and test CNN is provided (along with the data used for testing); with the hope that it will be useful to those conducting further studies into automatic music transcription approaching the stream level. Everything here is distributed with a CC-BY-4.0 license https://creativecommons.org/licenses/by/4.0/. If you would like to read my report then send me an email at: alialbarosa@gmail.com.
## Pre-Requisite Modules
- TensorFlow v2
- pyfftw         
- pretty_midi
- librosa
- matplotlib
- numpy
- h5py
- PIL
- sklearn
- scipy
- ffmpeg

# Generating Training Data
For training, 'instrument ascensions' were created: these are defined as the quarter note (crochet) ascension of a instrument from its lowest MIDI pitch, to its highest MIDI pitch. In the case of mixing instruments, an instrument ascension of one instrument was created for every note of the other instrument, to cover all possible pitch combinations. The "newCreation.ipynb" notebook can be used to generate single and dual label datasets for training with other instruments.

# Network Testing
The predictions from the network are tested via a piano roll sampled at 100Hz, giving a timing resolution of 10ms. The accuracy of the predicted piano-roll was measured by four activation metrics: "correct" (in time and pitch), "wrong" (correct in time but not in pitch), "false" (a note was found when there should have been none) and "missed" (there was no note found when there should have been one).
# Testing Data
The MIDI files used for the creation of the testing dataset are from a specific subset of the Lakh-MIDI dataset, developed by Colin Raffel (Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016) and available at: https://colinraffel.com/projects/lmd/, known as the "clean MIDI subset". This subset contains the music from songs of different genres in MIDI format. Eight of these MIDI files were adapted to include just two monophonic melodies, corresponding to the instruments that the network was trained to separate, and set to a tempo of 120BPM before being exported as WAV files; furthermore the audio of each of the instruments playing alone was exported in the same way to allow for more rigorous analysis of the performance of the networks. As a result there were three testing sets, each containing 35 minutes of music.
