import tensorflow as tf
# Hide GPU from visible devices, solve OOM issue by running test on CPU
#tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import h5py
import pretty_midi
keras = tf.keras

import makeDataset as getData
import midiCreationFuncs as midiMagic
import spectrogramFuncs as specMagic
import shutil
import pretty_midi as pm


def load_model(path):
# Loads a saved model (.ckpt format)
    model = keras.models.load_model(path)
    return model
            
def predict_one_inst(fPath, model, hop, window, reset = False):
# Get model predictions for a single instrument network, output is a pitch for every input
# within a folder
    specPath = fPath + 'Spectrograms' + 'F_' + str(window) + 'H_' + str(hop)
    if (os.path.isdir(specPath) == True) and (reset == True):
        shutil.rmtree(specPath)
        specMagic.create_spectrogram_set_from_wav_file(fPath, hopLength=hop, fwindow=window)

    if os.path.isdir(specPath) == False:
        specMagic.create_spectrogram_set_from_wav_file(fPath, hopLength=hop, fwindow=window) 

    specs = getData.spectrograms_to_arrays(specPath)
    prediction = model.predict(specs)
    notes = []
    for i in range(len(prediction)):
        notes.append(int(np.argmax(prediction[i])))
    return notes

def predict_two_insts(fPath, model, hop, window, reset = False):
# Get model predictions for a two instrument network, output is a pitch for each instrument
# for every input spectrogram within a folder. First output is guitar, second is trumpet.
    specPath = fPath + 'Spectrograms' + 'F_' + str(window) + 'H_' + str(hop)
    if (os.path.isdir(specPath) == True) and (reset == True):
        shutil.rmtree(specPath)
        specMagic.create_spectrogram_set_from_wav_file(fPath, hopLength=hop, fwindow=window)

    if os.path.isdir(specPath) == False:
        specMagic.create_spectrogram_set_from_wav_file(fPath, hopLength=hop, fwindow=window) 

    specs = getData.spectrograms_to_arrays(specPath)
    # get prediction
    predictions = model.predict(specs)
    guitar_notes = predictions[0]
    trumpet_notes = predictions[1]
    g, t = [], []
    for i in range(len(guitar_notes)):
        g.append(int(np.argmax(guitar_notes[i])))
        t.append(int(np.argmax(trumpet_notes[i])))
    return g,t

def predict_two_insts_w_split_specs(fPath, model, hop, window, reset = False):
# Gets model predictions for the split spectrogram format
    specPath = fPath + 'SplitSpectrograms' + 'F_' + str(window) + 'H_' + str(hop)
    if (os.path.isdir(specPath) == True) and (reset == True):
        shutil.rmtree(specPath)
        os.mkdir(specPath)
        specMagic.create_time_split_spectrograms_for_real_audio(fPath, hopLength=hop, fwindow=window) 
    if os.path.isdir(specPath) == False:
        os.mkdir(specPath)
        specMagic.create_time_split_spectrograms_for_real_audio(fPath, hopLength=hop, fwindow=window) 

    specs = getData.spectrograms_to_arrays(fPath + 'SplitSpectrogramsF_' + str(window) + 'H_' + str(hop))
    # get prediction
    predictions = model.predict(specs)
    guitar_notes = predictions[0]
    trumpet_notes = predictions[1]
    g, t = [], []
    for i in range(len(guitar_notes)):
        g.append(int(np.argmax(guitar_notes[i])))
        t.append(int(np.argmax(trumpet_notes[i])))
    return g,t

def get_piano_roll_metrics(predictions, midi_path, instrument, min_note_length): # 0 for guitar, 1 for trumpet
# Function to calculate performance metrics for an instruments output pitch prediction piano roll
# against the midi ground truth piano roll. If there are two instruments in the midi
# file the convention is that the first should be the guitar.
    inst = pm.Instrument(0, is_drum=False)
    song = pm.PrettyMIDI(midi_path + '.mid')
    start = 0
    notelength = min_note_length
    for i in range(len(predictions)):
        if predictions[i] == 128:
            start += notelength
        else:
            inst.notes.append(pm.Note(100, predictions[i], start, start+notelength))
            start += notelength

    fs = 100 # sample rate of the piano roll giving 10 ms time resolution
    end_time = song.get_end_time() # Use end time of midifile to get times
    times = np.arange(0, end_time, 1./fs)
    no_inst_flag = 0
    try:
        truth = song.instruments[instrument].get_piano_roll(fs, times=times)[40:89]
    except:
        dummy = pm.Instrument(0, is_drum=False)
        truth = dummy.get_piano_roll(fs, times=times)[40:89]
        no_inst_flag = 1

    pred = inst.get_piano_roll(fs, times=times)[40:89]

    correct = 0
    wrong = 0
    false = 0
    missed = 0

    total = 0

    for i in range(len(pred[0])): # Get time indexes
        predCol, truthCol = [],[]
        for predPitch, truthPitch in zip(pred, truth):
            predCol.append(float(predPitch[i]))
            try:
                truthCol.append(float(truthPitch[i]))
            except: 
                truthCol.append(0)
        # Normalise values
        predCol = [p/max(predCol) if p != 0 else 0 for p in predCol]
        truthCol = [t/max(truthCol) if t != 0 else 0 for t in truthCol]
        #print(predCol,truthCol)
        try:
            p = predCol.index(1)
            try:
                t = truthCol.index(1)
                if p == t:
                    correct += 1
                else:
                    wrong += 1
            except:
                false += 1       
        except: 
            try: 
                t = truthCol.index(1)
                missed += 1
            except:
                correct += 1
        total += 1
    
    if total == 0 and no_inst_flag == 1:
        metrics = [1,0,0,0]
        print(midi_path)
    elif total == 0:
        metrics = [0,0,0,1]
    else:
        metrics = [correct/total, wrong/total, false/total, missed/total]
    return metrics

## Test K fold networks - solo = 0,1,2 [both, guitar only, trumpet only]
def test_all_in_take(folderOfTakePath, testDataPath, hop, window, solo = 0, splits = 1): ## Must make sure that folder only contains window, hop and splits in one combination
# Tests a folder containing networks that have been trained k-fold times
    networks = os.listdir(folderOfTakePath)
    for network in networks:
        inst = None
        if '2_inst' in network or 'two_inst' in network:
            inst = 'Two'
        elif 'guitarsound' in network.lower():
            inst = 'Guitar'
        elif 'trumpetsound' in network.lower():
            inst = 'Trumpet'    

        best_fold = auto_test_network_folds(folderOfTakePath + '/' + network, inst, testDataPath, hop, window, solo, splits)
        print(network)
        print(best_fold[0])
        print('Guitar Metrics: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%'.format(best_fold[1][0][0]*100, best_fold[1][0][1]*100, best_fold[1][0][2]*100, best_fold[1][0][3]*100))
        print('Trumpet Metrics: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%'.format(best_fold[1][1][0]*100, best_fold[1][1][1]*100, best_fold[1][1][2]*100, best_fold[1][1][3]*100))
        print('Correct, Wrong, False, Missed\n')

# Takes average of all songs for each fold, returns the best folds results
def auto_test_network_folds(folderOfKs, inst, testDataPath, hop, window, solo, splits = 1):
# Tests all k-folds of a trained network
    folds = os.listdir(folderOfKs)
    songs = get_song_paths(testDataPath)
    highestCorrect = 0
    best_fold = ['', [[0,0,0,0],[0,0,0,0]] ] # Best fold is taken to be the one with the highest 'correct' average across instruments
    for fold in folds:
        model = load_model(folderOfKs + '/' + fold + '/Lowest_Loss.ckpt')
        song_av = [[0,0,0,0],[0,0,0,0]] # correct, wrong, false, missed
        for song in songs:
            #print(song)
            accuracies = test_model_on_song(song, model, hop, window, inst, solo, splits)
            for i in range(4): # add accuracies up
                song_av[0][i] += accuracies[0][i]
                song_av[1][i] += accuracies[1][i]
        for i in range(4):
            song_av[0][i], song_av[1][i] = song_av[0][i]/len(songs), song_av[1][i]/len(songs) # obtain average over all songs
        correct_av = (song_av[0][0] + song_av[1][0])/2
        if correct_av > highestCorrect:
            highestCorrect = correct_av
            best_fold[0] = fold
            best_fold[1][0], best_fold[1][1] = song_av[0], song_av[1]  

    del model
    gc.collect()
    return best_fold


def test_model_on_song(songFolder, model, hop, window, inst, solo, splits=1): ## Default assumes no splitting of spectrograms, returns array of size 2
# Tests a model on a given song (must be contained in a folder)
    files = os.listdir(songFolder)
    midiPath = None
    wav_path = None
    for filo in files:
        if '.mid' in filo:
            midiPath = songFolder + '/' + filo[:-4]
            continue
        if '.wav' in filo:
            ## Get predictions from wav file using model
            wav_path = songFolder + '/' + filo[:-4]
            #print(wav_path)

    if inst == 'Two':
        guitar, trumpet = predict_two_insts(wav_path, model, hop = hop, window = window)
        # Test predictions
        if solo == 0:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 1:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 2:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 1, (window*2)/(32768*splits))
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 0, (window*2)/(32768*splits))
        del guitar
        del trumpet
        gc.collect()
        return guitar_acc, trumpet_acc

    elif inst == 'Trumpet':
        trumpet = predict_one_inst(wav_path, model, hop = hop, window = window)
        if solo == 0:
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 1:
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 2:
            trumpet_acc = get_piano_roll_metrics(trumpet, midiPath, 0, (window*2)/(32768*splits))
        del trumpet
        gc.collect()
        return [0,0,0,0], trumpet_acc

    elif inst == 'Guitar':
        guitar = predict_one_inst(wav_path, model, hop = hop, window = window)
        if solo == 0:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
        elif solo == 1:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
        elif solo == 2:
            guitar_acc = get_piano_roll_metrics(guitar, midiPath, 1, (window*2)/(32768*splits))
        del guitar
        gc.collect()
        return guitar_acc, [0,0,0,0]

def get_song_paths(testDataPath):
# Gets paths of the folders containing songs
    songs = os.listdir(testDataPath)
    song_paths = []
    for song in songs:
        song_paths.append(testDataPath + '/' + song)
    return song_paths 

'''
def predictGuitarNotes(fName, model):
    # predict real guitar
    specMagic.create_spectrogram_set_from_wav_file('RealGuitar/'+ fName)
    specs = getData.spectrograms_to_arrays('RealGuitar/' + fName + 'Spectrograms')
    #print(specs)
    specs = np.array(specs)/255.0
    prediction = model.predict(specs)
    #print(prediction)
    for guess in prediction:
        note = pretty_midi.note_number_to_name(np.argmax(guess))
        if guess[np.argmax(guess)] >= 0.98:
            print(np.argmax(guess), note, guess[np.argmax(guess)]) 
def predict_guitar_pitch_onOff(fName, model, tempo = 120):
    # predict real guitar
    specMagic.create_spectrogram_set_from_wav_file('RealGuitar/'+ fName, tempo)
    specs = getData.spectrograms_to_arrays('RealGuitar/' + fName + '.wavSpectrograms')
    #print(specs)
    specs = np.array(specs)/255.0
    pitch_prediction = model.predict(specs)[0]
    onOff_prediction = model.predict(specs)[1]
    #print(prediction)
    i=0
    while i < len(pitch_prediction)-1:
        note = pretty_midi.note_number_to_name(np.argmax(pitch_prediction[i]))
        currentNote = np.argmax(pitch_prediction[i])
        c_act = onOff_prediction[i]
        # New Trick
        j = i + 1
        nextNote = np.argmax(pitch_prediction[j])
        n_act = onOff_prediction[j]
        while nextNote == currentNote and j < len(pitch_prediction)-1:
            if n_act > c_act*0.9:
                break
            else:
                j+=1
                nextNote = np.argmax(pitch_prediction[j])
                n_act = onOff_prediction[j]
        onOff_prediction[j] = 1
        print(currentNote, note, c_act, pitch_prediction[i][np.argmax(pitch_prediction[i])])
        i = j
    print('\n')
    for pitch, on in zip(pitch_prediction, onOff_prediction):
        pitch = np.argmax(pitch)
        note = pretty_midi.note_number_to_name(pitch)
        print(pitch, note, on)'''
