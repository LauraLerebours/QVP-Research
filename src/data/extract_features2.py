#Adapted from Alejandro Delgado's feature extraction for his beatbox onset detection software
#https://github.com/alejandrodl/beatbox-onset-detection
#in the data file
import os
import numpy as np


class extract_features_32():
    def __init__(self):
        self.num_extra_features = 4
        self.num_mel_coeffs = 14
        self.num_mel_bands = 40
        self.num_features = 32



