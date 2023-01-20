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

df = pd.DataFrame()

def filter_window(df, start, end, step):
        df_mlabel = []
        for x in range(start, end, step):
            df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x + step)]
            names = df_windowed['label']
            df_mlabel.append(df_windowed)
        return df_mlabel
    

def filter_label(row):
    return row['label'] in ['PHYCUV_M', 'PHYCUV_F']#'BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F']

    

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
       #MultiLabel Window Cutting
        df_mlabel = filter_window(df, 0, 60, 5)
 
print("Done!")    


#%% Merging mlabel list of dataframes to a single dataframe
v = pd.DataFrame()
v = pd.concat(df_mlabel,ignore_index=True)
#for i in range (len(df_mlabel)):
   # v=v.append(df_mlabel[i])
print("Done all in v")

# Print count of species founded    
print(v['label'].value_counts())


#%%
#Filtrar la especie
df_rois = df[df.apply(filter_label, axis=1)]

 #%%  
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
  

#%% save data frame
path_save_csv = "../SCRIPTS/TDS/PHYCUV/CSV/"
rois_fmt['fname_trim'] = rois_fmt.label + '_' + rois_fmt.index.astype(str).str.zfill(3)
rois_fmt.to_csv(path_save_csv+'rois_details.csv', index=False)


# %%  Trimming the audios 
import pandas as pd
from pydub import AudioSegment

path_save_aud = "../SCRIPTS/TDS/BOAALB/Audio_Trim/"
# Read the dataframe
df_csv = pd.read_csv('../SCRIPTS/TDS/rois_details.csv')
df_csv['min_t'] = df_csv['min_t'].mul(1000)
df_csv['max_t'] = df_csv['max_t'].mul(1000)
# Iterate through the dataframe
for index, row in df_csv.iterrows():
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
    audio_trimmed.export(path_save_aud + name + ".wav", format="wav")

print("All audio files have been trimmed successfully!")
# %%
