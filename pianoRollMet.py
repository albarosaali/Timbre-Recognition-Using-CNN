import pitchDetectionNetTesting as testFunc
import midiCreationFuncs as midiMagic
import matplotlib.pyplot as plt
import pretty_midi as pm
import librosa
import numpy as np
import os
import re
import gc
'''
## Test K fold networks - solo = 0,1,2 [both, guitar only, trumpet only]
def test_all_in_take(folderOfTakePath, testDataPath, hop, window, solo = 0, splits = 1): ## Must make sure that folder only contains window, hop and splits in one combination
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
    folds = os.listdir(folderOfKs)
    songs = get_song_paths(testDataPath)
    highestCorrect = 0
    best_fold = ['', [[0,0,0,0],[0,0,0,0]] ] # Best fold is taken to be the one with the highest 'correct' average across instruments
    for fold in folds:
        model = testFunc.load_model(folderOfKs + '/' + fold + '/Lowest_Loss.ckpt')
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
        guitar, trumpet = testFunc.predict_two_insts(wav_path, model, hop = hop, window = window)
        # Test predictions
        if solo == 0:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 1:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 2:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 1, (window*2)/(32768*splits))
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 0, (window*2)/(32768*splits))
        del guitar
        del trumpet
        gc.collect()
        return guitar_acc, trumpet_acc

    elif inst == 'Trumpet':
        trumpet = testFunc.predict_one_inst(wav_path, model, hop = hop, window = window)
        if solo == 0:
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 1:
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 1, (window*2)/(32768*splits))
        elif solo == 2:
            trumpet_acc = testFunc.get_piano_roll_metrics(trumpet, midiPath, 0, (window*2)/(32768*splits))
        del trumpet
        gc.collect()
        return [0,0,0,0], trumpet_acc

    elif inst == 'Guitar':
        guitar = testFunc.predict_one_inst(wav_path, model, hop = hop, window = window)
        if solo == 0:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
        elif solo == 1:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 0, (window*2)/(32768*splits))
        elif solo == 2:
            guitar_acc = testFunc.get_piano_roll_metrics(guitar, midiPath, 1, (window*2)/(32768*splits))
        del guitar
        gc.collect()
        return guitar_acc, [0,0,0,0]

def get_song_paths(testDataPath):
    songs = os.listdir(testDataPath)
    song_paths = []
    for song in songs:
        song_paths.append(testDataPath + '/' + song)
    return song_paths
'''
print("Guitar + Trumpet\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Guitar + Trumpet', hop = 512, window = 2048, solo = 0)
print("Guitar\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Guitar', hop = 512, window = 2048, solo = 1)
print("Trumpet\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Trumpet', hop = 512, window = 2048, solo = 2)

