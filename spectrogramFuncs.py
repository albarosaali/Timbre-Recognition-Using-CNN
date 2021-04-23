import os
import shutil
import numpy as np
import pyfftw
import pydub
from pydub import AudioSegment as AS
import scipy
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import ffmpeg
import librosa
import multiprocessing
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

def make_mono_downsample(filename):
    # Take song and make sure it is mono
    audio,_ = librosa.load(filename, sr=32768)
    audio = librosa.to_mono(audio)
    return audio

def get_split_time(tempo,fs = 44100):
    # For quarter note resolution clips
    spb = 60/(tempo)
    return spb

def get_32th_splits(tempo,fs = 44100):
    spb = 60/(tempo*2*2*2)
    return spb
'''
def split_audio(filename, split):
    # Make sure file is mono
    #make_mono(filename)
    # Find path to current folder
    path = os.getcwd() + '/' + filename + 'Clips'
    # If new folder already exists then delete it
    if os.path.isdir(path) == True:
        shutil.rmtree(path)
    os.mkdir(path)
    # Prepare to split audio
    song = AS.from_wav(filename)
    start = 0
    i = 0
    # Convert split to ms
    split = split*1000
    while start < len(song):
        clip = song[start:start + split]
        clip.export(path + '/clip'+ str(i) + '.wav', format = "wav")
        start += split
        i += 1
        #print(len(song),start)
    return path
    '''

def sample_frames(data, fwindow = 4096, overlap = None):
# Get number of frames per frame(rounded down)
# Assume frame size around 20 ms
# Approx between 64th and 128th note at 120 bpm
    frameSize = fwindow 
    if overlap == None:
        # 50%
        overlap = 0.5
    overlap = frameSize - int(overlap*frameSize)
    start = 0
    frames = []
    # Split data into frames of size = frameSize
    while start + frameSize < len(data):
        frames.append(data[start:start+frameSize])
        start += overlap
    # Apply Kaiser Window, beta = 8 to each sample
    window = np.kaiser(len(frames[0]),8)
    for i in range(len(frames)):
        try:
            frames[i] = frames[i]*window
        except:
            window = np.kaiser(len(frames[i]),8)
            frames[i] = frames[i]*window
    return frames 

def obtain_FFTs(frames):
# Apply fourier transform to each frame
    FFTs = []
    a = pyfftw.empty_aligned(len(frames), dtype='complex128')
    b = pyfftw.empty_aligned(len(frames), dtype='complex128')
    fft_object = pyfftw.FFTW(a,b)
    for frame in frames:
        try:
            a[:] = frame
        except:
            a = pyfftw.empty_aligned(len(frame), dtype='complex128')
            b = pyfftw.empty_aligned(len(frame), dtype='complex128')
            fft_object = pyfftw.FFTW(a,b)
            a[:] = frame
        fft = fft_object()
        N = len(a)
        FFTs.append(abs((fft[:int(N//2)])**2))
    return FFTs

def FFT_frequencies(frame_length, fs = 44100):
    deltaF = (fs/2)/(frame_length)
    frequencies = np.arange(0,int(fs/2), deltaF)
    return frequencies

def plot_spectrogram(spectro):
    plt.figure(dpi = 100, figsize = [10,8])
    plt.rcParams.update({'font.size': 24})
    #ax = plt.subplot()
    # xlabel for litreview
    x = np.arange(0,17.6, step=2.5)
    plt.imshow(spectro, aspect='auto', origin = 'lower', extent = [0,17.5,1,22050], cmap = 'viridis')
    #plt.colorbar(spec)
    plt.yscale('log')
    plt.ylim(20,22050)
    plt.xlabel('Time (s)')
    plt.xticks(x,['0.0','2.5','5.0','7.5','10.0','12.5','15.0','17.5'])
    plt.ylabel('Frequency (Hz)')
    plt.show()

def save_spectrogram(spectro,filename):
    fig = plt.figure(dpi = 100, figsize = [1.28,5.12], frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(spectro, aspect='auto', origin = 'lower', extent = [0,125,1,16384], cmap = 'viridis')
    plt.yscale('log')
    plt.ylim(20, 16384)
    filename += '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)

def save_split_spectrogram(spectro,filename,num):
    fig = plt.figure(dpi = 100, figsize = [1.28,5.12], frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(spectro, aspect='auto', origin = 'lower', extent = [0,128,1,16384], cmap = 'viridis')
    plt.yscale('log')
    plt.ylim(20, 16384)
    for split in range(0,128,16):
        plt.xlim(split, split+16)
        splitSave = filename + str(num) + '.png' 
        plt.savefig(splitSave, bbox_inches='tight', pad_inches = 0)
        num += 1
    plt.close(fig)
    return num

def create_spectrogram_set_from_wav_file(filename, hopLength = 2048, fwindow = 4096, reset = False):
    # Get filepath
    #path = os.getcwd() + '/' + filename + '.wav'
    path = filename + '.wav'
    audio = make_mono_downsample(path)
    # Make path for spectrograms
    specPath = path[:-4] + 'Spectrograms' + 'F_' + str(fwindow) + 'H_' + str(hopLength)
    if os.path.isdir(specPath) == True and reset == False:
        return
    os.mkdir(specPath) 
    i = 0
    j = 0    
    overlap = 1 - hopLength/fwindow
    frames = sample_frames(audio, fwindow = fwindow,  overlap=overlap)
    '''
    transformed = obtain_FFTs(frames[0:1])
    transformed = np.array(transformed).T
    logTrans = np.log10(transformed)
    specLoc = specPath + '/spec'
    save_spectrogram(logTrans, specLoc)
    #plot_spectrogram(logTrans)
    '''
# for fwindow = 2048 -> spec = 4096 samples = 0.125s, 8Hz @ 32768 Hz fs
# for fwindow = 4096 -> spec = 8192 samples = 0.25s, 4Hz @ 32768 Hz fs
    while i < len(frames):
        transformed = obtain_FFTs(frames[i:i + int((fwindow/hopLength + 1))]) # adjust for hop
        transformed = np.array(transformed).T
        logTrans = np.log10(transformed)
        specLoc = specPath + '/spec' + str(j)
        save_spectrogram(logTrans, specLoc)
        i+= int(fwindow/hopLength)*2 # adjust for hop
        j+=1
    #print("Spectrograms Created")

def create_time_split_spectrograms_for_real_audio(filename, hopLength = 2048, fwindow = 4096, reset = False):
    # Get filepath
    #path = os.getcwd() + '/' + filename + '.wav'
    path = filename +'.wav'
    audio = make_mono_downsample(path)
    # Make path for spectrograms
    specPath = path[:-4] + 'SplitSpectrograms' + 'F_' + str(fwindow) + 'H_' + str(hopLength)
    if os.path.isdir(specPath) == True:
        shutil.rmtree(specPath)
    os.mkdir(specPath) 
    i = 0
    j = 0   
    #fName = path
    overlap = 1 - hopLength/fwindow
    frames = sample_frames(audio, fwindow = fwindow,  overlap=overlap)
    '''
    transformed = obtain_FFTs(frames[0:1])
    transformed = np.array(transformed).T
    logTrans = np.log10(transformed)
    specLoc = specPath + '/spec'
    save_spectrogram(logTrans, specLoc)
    #plot_spectrogram(logTrans)
    '''
    while i < len(frames):
        transformed = obtain_FFTs(frames[i:i + int((fwindow/hopLength + 1))]) # adjust for hop
        transformed = np.array(transformed).T
        logTrans = np.log10(transformed)
        specLoc = specPath + '/spec'
        j = save_split_spectrogram(logTrans, specLoc,j)
        i+= int(fwindow/hopLength)*2 # adjust for hop

'''
def sample_32thClips(filename, tempo, overlap = None):
    fs, data = wav.read(filename)
    # Take framesize = 64th Notes
    # therefore 50% overlap -> 128th Notes
    # therefore 3 combined -> spectrogram of a 32th note
    frameSize = int((60/(tempo*2*2*2*2)*fs))
    if overlap == None:
        overlap = 0.5
    overlap = 1 - int(overlap*frameSize)
    start = 0
    count = 0
    frames = []
    while (start + frameSize) < len(data):
        if count < 3:
            frames.append(data[start:start+frameSize])
            start += overlap
            count += 1
        else:
            start += overlap
            count = 0
    # Apply Kaiser Window, beta = 8 to each sample
    window = np.kaiser(len(frames[0]), 8)
    for i in range(len(frames)):
        try:
            frames[i] = frames[i]*window
        except:
            window = np.kaiser(len(frames[i]), 8)
            frames[i] = frames[i]*window
    return frames

def sample_16thClips(filename, tempo, overlap = None):
    fs, data = wav.read(filename)
    # Take framesize = 32th Notes
    # therefore 50% overlap -> 64th Notes
    # therefore 3 combined -> spectrogram of a 16th note
    frameSize = int((60/(tempo*2*2*2)*fs))
    if overlap == None:
        overlap = 0.5
    overlap = frameSize - int(overlap*frameSize) 
    start = 0
    count = 0
    frames = []
    while (start + frameSize) < len(data):
        if count < 3:
            frames.append(data[start:start+frameSize])
            start += overlap
            count += 1
        else:
            start += overlap
            count = 0
    # Apply Kaiser Window, beta = 8 to each sample
    window = np.kaiser(len(frames[0]), 8)
    for i in range(len(frames)):
        try:
            frames[i] = frames[i]*window
        except:
            window = np.kaiser(len(frames[i]), 8)
            frames[i] = frames[i]*window
    return frames
'''
'''
def obtain_FFTs(samples):
    # Apply fourier transform to each sample
    FFTs = []
    a = pyfftw.empty_aligned(len(samples[0]), dtype='complex128')
    b = pyfftw.empty_aligned(len(samples[0]), dtype='complex128')
    fft_object = pyfftw.FFTW(a,b)
    for sample in samples:
        try:
            a[:] = sample
        except:
            a = pyfftw.empty_aligned(len(sample), dtype='complex128')
            b = pyfftw.empty_aligned(len(sample), dtype='complex128')
            fft_object = pyfftw.FFTW(a,b)
            a[:] = sample
        fft = fft_object()
        N = len(sample)
        FFTs.append(abs(fft[:int(N//2)]))
    return FFTs

def save_spectrogram_for_guitar(spectro,filename):
    fig = plt.figure(dpi = 100, figsize = [1.28,5.12], frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(spectro, aspect='auto', origin = 'lower', extent = [0,500,1,22050])
    plt.yscale('log')
    plt.ylim(70, 22050) # Changed (20,22050)
    filename += '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)

def create_spectrogram_set_from_wav_file(filename, tempo, guitar = False):
    make_mono(filename)
    # Add sample rate arg if not 44100
    split = get_32th_splits(tempo)
    # Create new wav files of splits
    path = split_audio(filename, split)
    # Make path for spectrograms
    specPath = path + '/spectrograms'
    if os.path.isdir(specPath) == True:
        shutil.rmtree(specPath)
    os.mkdir(specPath)    
    i = 0
    while True:
        fName = path + '/clip' + str(i) + '.wav'
        try:
            # Add arg here to change overlap
            samples = sample_frames(fName, tempo)
        except:
            break
        transformed = obtain_FFTs(samples[0])
        transformed = np.array(transformed).T
        logTrans = np.log10(transformed)
        specLoc = specPath + '/spec' + str(i)
        if guitar == True:
            save_spectrogram_for_guitar(logTrans, specLoc)
        else: save_spectrogram(logTrans, specLoc)
        i+=1
'''