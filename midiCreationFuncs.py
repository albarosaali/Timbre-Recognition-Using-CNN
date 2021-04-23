import pretty_midi as pm
import numpy as np
from numpy import asarray
import librosa.display
import matplotlib.pyplot as plt 
import h5py 
# piano 21-108
# guitar pn = 27-30 notes: 40-88
# trump 56 notes: 54-86
def create_instrument_ascension(filename, directory, noteLength=0.5, instrument_prog_num = 0, lowestNote = 21, highestNote = 108, noteRepeats = 1, totalRepeats = 1):
    # Default is full range of a piano playing quarter notes
    track = pm.PrettyMIDI(initial_tempo=120)
    inst = pm.Instrument(program = instrument_prog_num, is_drum = False)
    track.instruments.append(inst)
    # note velocity
    velocity = 100
    pitches = np.arange(lowestNote, (highestNote+1), step=1)
    # Default note length is a quarter note
    noteStart = 0
    storeRepeat = noteRepeats
    while totalRepeats > 0:
        for pitch in pitches:
            while noteRepeats > 0:
                inst.notes.append(pm.Note(velocity, pitch, noteStart, noteStart+noteLength))
                noteStart += noteLength
                noteRepeats -= 1
            noteRepeats = storeRepeat
        totalRepeats -= 1
    # Save midiFile
    track.write(directory + '/' + filename + '.mid')
    return track

# Start off for guitar and trumpet [0,1]
# Labels ascensions in two label format
def create_inst_label(path, noteLength, shortestLen):
    info = pm.PrettyMIDI(path, initial_tempo=120)
    guit = []
    trump = []
    if len(info.instruments) == 1:
        if ("Guitar" in pm.program_to_instrument_name(info.instruments[0].program)):
            for note in info.instruments[0].notes:
                i=0
                while i*shortestLen < noteLength:
                    guit.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                    trump.append([0]*128 + [1])
                    i += 1
        elif ("Trumpet" in pm.program_to_instrument_name(info.instruments[0].program)):
            for note in info.instruments[0].notes:
                i=0
                while i*shortestLen < noteLength:
                    guit.append([0]*128 + [1])
                    trump.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                    i += 1
    elif (len(info.instruments) == 2):
        guitar = info.instruments[0].notes
        trumpet = info.instruments[1].notes
        for note in trumpet:
            i = 0
            while i*shortestLen < noteLength:
                trump.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                i += 1
        for note in guitar:
            i = 0
            while i*shortestLen < noteLength:
                guit.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                i += 1
    return guit, trump

# Start off for guitar and trumpet [0,1]
def create_sole_inst_label(path, noteLength, shortestLen, label_instrument):
    info = pm.PrettyMIDI(path)
    inst = []
    if len(info.instruments) == 1:
        if ("Guitar" in pm.program_to_instrument_name(info.instruments[0].program)):
            if label_instrument != 'Guitar':
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*128 + [1])
                        i += 1
            else:            
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                        i += 1
        elif ("Trumpet" in pm.program_to_instrument_name(info.instruments[0].program)):
            if label_instrument != 'Trumpet':
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*128 + [1])
                        i += 1
            else:            
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                        i += 1
        elif ("Piano" in pm.program_to_instrument_name(info.instruments[0].program)):
            if label_instrument != 'Piano':
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*128 + [1])
                        i += 1
            else:            
                for note in info.instruments[0].notes:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                        i += 1
    elif (len(info.instruments) == 2):
        if label_instrument == 'Guitar':
            guitar = info.instruments[0].notes
            i = 0
            for note in guitar:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                        i += 1
        if label_instrument == 'Trumpet':
            trumpet = info.instruments[1].notes
            i = 0
            for note in trumpet:
                    i=0
                    while i*shortestLen < noteLength:
                        inst.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                        i += 1
    return inst

def create_piano_roll(noteList, instrument, min_note_length):
    inst = pm.Instrument(0, is_drum=False, name=instrument)
    start = 0
    notelength = min_note_length   
    for i in range(len(noteList)):
        if noteList[i] == 128:
            start += notelength
        else:
            inst.notes.append(pm.Note(100, noteList[i], start, start+notelength))
            start += notelength
    if len(inst.notes) != 0:
        librosa.display.specshow(inst.get_piano_roll(100)[40:89], hop_length=1, sr=500, x_axis='time',
        y_axis='cqt_note', fmin=pm.note_number_to_hz(40), fmax=pm.note_number_to_hz(88))
    else: 
        print('No instrument detected')
    return inst

def create_midi_playback(instrument, fname, tempo):
    playback = pm.PrettyMIDI(initial_tempo = tempo)
    playback.instruments.append(instrument)
    playback.write('ValidationData(120)/DetectedPlayback/' + fname + '.mid')

# Assume guitar is inst = 0, trumpet is inst = 1
def create_test_labels_one_inst(midi_file, min_note_length, instrument = None):
    info = pm.PrettyMIDI(midi_file + '.mid', initial_tempo=120)
    labels = []
    if instrument == None or len(info.instruments) == 1:
        inst = 0
    elif instrument == 'Guitar':
        inst = 0
    elif instrument == 'Trumpet' or instrument == 'Piano':
        inst = 1
    
    inst = info.instruments[inst] # need to take into account gaps between note ends and starts
    previous_note_end_time = None

    ## Must take into account notes that do not start right at the beginning
    if inst.notes[0].start > 0:
        i = 0
        while i * min_note_length < inst.notes[0].start: # gap
            labels.append([0]*128 + [1])
            i+=1

    for note in inst.notes:
        note_length = note.end - note.start
        if previous_note_end_time == None:
            i = 0
            while i * min_note_length < note_length:
                labels.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                i+=1
            previous_note_end_time = note.end
            #print('1st')

        elif note.start - previous_note_end_time > 0.1: # means there is a gap
            #print(note.start, previous_note_end_time)
            #print('gap')
            i = 0
            while i * min_note_length < note.start - previous_note_end_time: # gap
                labels.append([0]*128 + [1])
                i+=1
            previous_note_end_time = note.end

        else:
            i = 0
            while i * min_note_length < note_length:
                labels.append([0]*(note.pitch) + [1] + [0]*(127-note.pitch) + [0])
                i+=1
            previous_note_end_time = note.end

    return labels


#noteLength = 0.125
#pitches = create_pitch_labels_from_midi_16thFrames('16thClips/Guitar16Asc.mid', 0.125)
#test = create_on_off_labels(len(pitches), 0.125)

#for i in range(0,12):
#    print(np.argmax(pitches[i]), test[i])


'''
labels = create_pitch_labels_from_midi('midiFiles/120QuarterNotePianoAscension.mid')
#track = create_instrument_ascension('120EigthNotePianoAscension', 'midiFiles', 0.25)
#labels = create_labels_from_midi('midiFiles/120QuarterNotePianoAscension.mid')
print(labels[0])
print(len(labels[0]))
print(labels[0].index(1))
#print(labels)
#print(track.instruments[0].notes[0].velocity)
'''


#create_instrument_ascension('4Asc','TrainingRawData(120)/Piano', noteLength=0.5, instrument_prog_num=0, lowestNote=21, highestNote=108, noteRepeats=1, totalRepeats=1)
'''

track = pm.PrettyMIDI(initial_tempo=120)
guitar = pm.Instrument(program = 27, is_drum=False)
trumpet = pm.Instrument(program = 56, is_drum = False)
track.instruments.append(guitar)
track.instruments.append(trumpet)
# note velocity
velocity = 100
guit_pitches = np.arange(40, 88+1, step=1)
trump_pitches = np.arange(54, 86+1, step=1)
# Default note length is a quarter note = 0.5s
noteStart = 0
for pitch in guit_pitches: 
    for tPitch in trump_pitches:
        guitar.notes.append(pm.Note(velocity, pitch, noteStart, noteStart + 0.5))
        trumpet.notes.append(pm.Note(velocity, tPitch, noteStart, noteStart + 0.5))
        noteStart += 0.5
# Save midiFile
track.write('TrainingRawData(120)/Guitar+Trumpet' + '/' + '4Asc' + '.mid')'''