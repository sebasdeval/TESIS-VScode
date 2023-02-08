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
import glob
from os import walk
import sphinx
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import wave

#%%
# Set main variables
target_fs = 24000  # target fs of project
wl = 5  # Window length for formated rois
path_annot = '../ANNOTATIONS_INTC41/INCT41/'  # location of bbox annotations
path_audio = '../AUDIO_INTC41/INCT41/'  # location of raw audio
path_save = '../train_dataset/'  # location to save new samples




nombre = next(walk(r"C:\Users\sebas\Documents\GitHub\TESIS-VScode\ANNOTATIONS_INTC41\INCT41"), (None, None, []))[2]  # [] if no file
# anotate = next(walk(r"C:\Users\sebas\OneDrive\ESCRITORIO 2022-2\TESIS\test_files\ANNOTATIONS_INTC41\INCT41"), (None, None, []))[2]

#%%
df = pd.DataFrame()


def filter_window(df, start, end, step):
    df_mlabel = []
    fname_lists = {}
    for x in range(start, end, step):
        df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x + step)]
        df_mlabel.append(df_windowed)
        for fname, group in df_windowed.groupby('fname'):
            if fname not in fname_lists:
                fname_lists[fname] = [group]
            else:
                fname_lists[fname].append(group)
    return df_mlabel, fname_lists
    

def filter_label(row):
    return row['label'] in ['PHYCUV_M', 'PHYCUV_F','BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F']

    
def save_fname_lists(fname_lists, save_path):
    for fname, dfs in fname_lists.items():
        dir_path = os.path.join(save_path, fname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i, df in enumerate(dfs):
            file_path = os.path.join(dir_path, f"{fname}_{i}.csv")
            df.to_csv(file_path, index=False)

#Load multiple annotations from directory
for i in range (len(nombre)): 
   # print(nombre[i])
    flist = glob.glob(path_annot + nombre[i])
    for fname in flist:
        df_aux = util.read_audacity_annot(fname)
        df_aux['fname'] = os.path.basename(fname).replace('.txt', '.wav')
        df_aux = df_aux.drop(columns=['min_f', 'max_f'])
        df = pd.concat([df,df_aux],ignore_index=True)
        #df = df.append(df_aux)
        df.reset_index(inplace=True, drop=True)
      #  print(df)
       #MultiLabel Window Cutting
        df_mlabel, fname_lists = filter_window(df,0,60,5)

#saving files
 #%%       
save_fname_lists(fname_lists, '../SCRIPTS/TDL/PHYCUV/CSV/')        
        
       
print("Done!")  

#%% Merging mlabel list of dataframes to a single dataframe to count numbers of label  on audios
v = pd.DataFrame()
v = pd.concat(df_mlabel,ignore_index=True)
#for i in range (len(df_mlabel)):
   # v=v.append(df_mlabel[i])
print("Done all in v")

# Print count of species founded    
print(v['label'].value_counts())


#%%
#Filtrar la especie
# df_rois = df[df.apply(filter_label, axis=1)].reset_index()


#%% TRIMEANDO AUDIOS

# Set the length of each audio chunk in seconds
chunk_length = 5

# Specify the source and destination directories for the audio files
src_dir = '../AUDIO_INTC41/INCT41/'
dst_dir = '../SCRIPTS/TDL/PHYCUV/AUDIO_TRIM/'

# Loop through the audio files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith(".wav"):
        # Open the audio file
        with wave.open(os.path.join(src_dir, filename), 'rb') as audio_file:
            # Get the number of frames in the audio file
            num_frames = audio_file.getnframes()
            # Get the frame rate of the audio file
            frame_rate = audio_file.getframerate()
            # Calculate the number of chunks in the audio file
            num_chunks = num_frames // (frame_rate * chunk_length)
            
            # Make a folder for each audio file
            file_dir = os.path.join(dst_dir, filename.split(".")[0])
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            
            # Loop through the chunks of audio
            for chunk_index in range(num_chunks + 1):
                # Create a new audio file for each chunk
                chunk_filename = f"{filename.split('.')[0]}_{chunk_index}.wav"
                chunk_file_path = os.path.join(file_dir, chunk_filename)
                with wave.open(chunk_file_path, 'wb') as chunk_file:
                    chunk_file.setnchannels(audio_file.getnchannels())
                    chunk_file.setsampwidth(audio_file.getsampwidth())
                    chunk_file.setframerate(audio_file.getframerate())
                    chunk_start = chunk_index * chunk_length * frame_rate
                    chunk_end = chunk_start + chunk_length * frame_rate
                    chunk_file.writeframes(audio_file.readframes(chunk_end - chunk_start))
print("ALL AUDIOS TRIMMED SUCCESFULLY")



#%% GENERANDO ESPECTROGRAMAS EN CADA FOLDER DE AUDIOS

def generate_spectrogram(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path)
                D = librosa.stft(y)
                S = librosa.amplitude_to_db(abs(D))

                plt.axis('off')
                plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                librosa.display.specshow(S, sr=sr, cmap='inferno')
                plt.savefig(file_path[:-4] + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()

path = '../SCRIPTS/TDL/PHYCUV/AUDIO_TRIM/'
generate_spectrogram(path)

#%%
import os
import librosa
import matplotlib.pyplot as plt
import multiprocessing as mp

def generate_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(abs(D))

    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(S, sr=sr, cmap='inferno')
    plt.savefig(file_path[:-4] + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_spectrogram_parallel(path):
    pool = mp.Pool(mp.cpu_count())

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                pool.apply_async(generate_spectrogram, args=(file_path,))

    pool.close()
    pool.join()

path = '../SCRIPTS/TDL/PHYCUV/AUDIO_TRIM/'
generate_spectrogram_parallel(path)