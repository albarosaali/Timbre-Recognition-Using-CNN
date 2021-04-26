import pitchDetectionNetTesting as testFunc
import midiCreationFuncs as midiMagic
import matplotlib.pyplot as plt
import pretty_midi as pm
import librosa
import numpy as np
import os
import re
import gc

print("Guitar + Trumpet\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Guitar + Trumpet', hop = 512, window = 2048, solo = 0)
print("Guitar\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Guitar', hop = 512, window = 2048, solo = 1)
print("Trumpet\n")
testFunc.test_all_in_take('SavedWeights/Test', 'C:/Users/alial/Documents/Final year project/Test Data 120/Trumpet', hop = 512, window = 2048, solo = 2)

