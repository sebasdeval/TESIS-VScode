#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize audio annotation with scikit-maad

@author: jsulloa
"""

import matplotlib.pyplot as plt
from maad import sound, util

#%% Set variables
fname_audio = './audio2/INCT20955_20191012_201500.wav'
fname_annot = './annotations2/INCT20955_20191012_201500.txt'

#fname_audio = './audio/LEAR01_20190911_013000.wav'
#fname_annot = './annotations/LEAR01_20190911_013000.txt'

#%% load data
s, fs = sound.load(fname_audio)
df_rois = util.read_audacity_annot(fname_annot)

# Visualize annotation
Sxx, tn, fn, ext = sound.spectrogram(s, fs=fs, nperseg=512,
                                     noverlap=256)
Sxx = util.power2dB(Sxx, db_range=60, db_gain=20)
df_rois = util.format_features(df_rois, tn, fn)

fig, ax = plt.subplots(1, figsize=(15,5))
util.plot_spectrogram(Sxx, ext, log_scale=False, colorbar=False, ax=ax)
util.overlay_rois(Sxx, df_rois, fig=fig, ax=ax)
