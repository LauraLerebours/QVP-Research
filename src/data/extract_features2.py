#Adapted from Alejandro Delgado's feature extraction for his beatbox onset detection software
#https://github.com/alejandrodl/beatbox-onset-detection
#in the data file
# import os
# import numpy as np
# import tensorflow as flow


# class extract_features_32():
#     def __init__(self):
#         self.num_extra_features = 4
#         self.num_mel_coeffs = 14
#         self.num_mel_bands = 40
#         self.num_features = 32


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
