# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 14:50:41 2022

@author: Michael || Sebastian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset

Audio recordings come into monophonic recordings. Manual annotations come into 
time-frequency segmentation performed using audacity or raven. There is one annotation
file per audio recording.

1. Transform bounding box annotation data to fixed size samples

"""
#%%
import numpy as np  
import pandas as pd
from maad import sound, util
from utils import roi2windowed, find_file
import os
import glob
from os import walk

import sys  #para sphinx
import pathlib  #para sphinx

# Set main variables
target_fs = 24000  # target fs of project
wl = 5  # Window length for formated rois
path_annot = "../ANNOTATIONS_INTC41/INCT41/"  # location of bbox annotations
path_audio = "../AUDIO_INTC41/INCT41/"  # location of raw audio
path_save = "../SCRIPTS/TDS/"  # location to save new samples




nombre = next(walk('/workspaces/TESIS-VScode/ANNOTATIONS_INTC41/INCT41'), (None, None, []))[2]  # [] if no file
anotate = next(walk('/workspaces/TESIS-VScode/ANNOTATIONS_INTC41/INCT41'), (None, None, []))[2]
df_mlabel= []
df = pd.DataFrame()

#Load multiple annotations from directory
for i in range (len(nombre)): 
    print(nombre[i])
    flist = glob.glob(path_annot + nombre[i])
    for fname in flist:
        df_aux = util.read_audacity_annot(fname)
        df_aux['fname'] = os.path.basename(fname).replace('.txt', '.wav')
        df_aux = df_aux.drop(columns=['min_f', 'max_f'])
        df = pd.concat([df,df_aux],ignore_index=True)
        #df = df.append(df_aux)
        df.reset_index(inplace=True, drop=True)
        print(df)
        
        
# Select vocalizations of a single species   
    df_rois = df.loc[(df.label=='BOAALB_M')|(df.label=='BOAALB_F'), :]# |(df.label=='BOALUN_F') | (df.label=='BOALUN_M')|(df.label=='PHYCUV_M')|(df.label=='PHUCUV_F')|(df.label=='DENCRU_M')|(df.label=='DENCRU_F')|(df.label=='PITAZU_F')|(df.label=='PITAZU_M'), :]    

   
# format rois to a fixed window
    rois_fmt = pd.DataFrame()
    for idx, roi in df_rois.iterrows():
        roi_fmt = roi2windowed(wl, roi)
        rois_fmt = pd.concat([rois_fmt,roi_fmt],ignore_index=True)
        #rois_fmt = rois_fmt.append(roi_fmt)
        
    rois_fmt.reset_index(inplace=True, drop=True)
    #print(rois_fmt)
    rois_fmt=rois_fmt.dropna(subset=['fname'])
    df_filt=(df[df["max_t"] <= 5])
    
    #MultiLabel Window Cutting
    for x in range(0, 60 ,5):
            print (x)
            df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x+5)]
            names = df_windowed['label']
            df_mlabel.append(df_windowed)
          #  print ('-------')
          #  print (names)
           # print(df_windowed)
        #Cada posición de la lista corresponde a una ventana de datos segmentados de acuerdo a wl comprendido en dataframes
        #Each position on mlabel list correspond to a data segmented window comprehended in dataframes 
    #print(df_mlabel)   
print("Done!")       

#%% Merging mlabel list of dataframes to a single dataframe
v = pd.DataFrame()
v = pd.concat(df_mlabel,ignore_index=True)
#for i in range (len(df_mlabel)):
   # v=v.append(df_mlabel[i])
print("Done all in v")

#%% Print count of species founded    
print(v['label'].value_counts())

###########
#%% Load audio, resample, trim, normalize and write to disk
#for idx_row, roi_fmt in rois_fmt.iterrows():
 #   print(idx_row+1, '/', len(rois_fmt))
 #   full_path_audio = find_file(roi_fmt.fname, path_audio)[0]
 #   fname_audio = os.path.join(full_path_audio)
 #   s, fs = sound.load(fname_audio)
 #   s = sound.resample(s, fs, target_fs)
 #   s_trim = sound.trim(s, target_fs, roi_fmt.min_t, roi_fmt.max_t, pad=True)

 #   fname_save = os.path.join(path_save, roi_fmt.label+'_'+str(idx_row).zfill(3)) 
 #   sound.write(fname_save+'.wav', fs=target_fs, data=s_trim)

#%% save data frame
rois_fmt['fname_trim'] = rois_fmt.label + '_' + rois_fmt.index.astype(str).str.zfill(3)
rois_fmt.to_csv(path_save+'rois_details.csv', index=False)


# %%
import pandas as pd
from pydub import AudioSegment

# Read the dataframe
df = pd.read_csv('../SCRIPTS/TDS/rois_details.csv')

# Iterate through the dataframe
for index, row in df.iterrows():
    # Get the path of the audio file
    audio_path = "../AUDIO_INTC41/INCT41/"
    # Get the start and end time for trimming
    start_time = row['min_t']
    end_time = row['max_t']
    name = row['fname_trim']
    f_name = row["fname"]
    
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path + f_name)
    # Trim the audio
    audio_trimmed = audio[start_time:end_time]
    # Save the trimmed audio
    audio_trimmed.export(path_save + name, format="wav")

print("All audio files have been trimmed successfully!")
# %%
