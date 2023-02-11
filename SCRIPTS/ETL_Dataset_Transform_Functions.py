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

This code is a Python script for processing audio files and generating spectrograms and annotations.
The script starts by importing necessary libraries including Numpy, Pandas, Librosa, Matplotlib, etc.
It then sets some global variables such as the target frequency, window length, paths to audio and annotations, etc.
The code includes several functions: filter_window filters a dataframe based on start, end, and step time values and returns a list
of dataframes and a dictionary of dataframes grouped by the file name. filter_label filters dataframe rows based on specific labels. 
save_fname_lists saves data in the fname_lists dictionary to CSV files.

"""
#%%
"""Libraries used.
"""
# Loading libraries
import numpy as np  
import pandas as pd
from maad import sound, util
import glob
from os import walk
import sphinx
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import wave

print ("Done loading libraries")

#%%
# Set main variables
target_fs = 24000  # target fs of project
wl = 5  # Window length for formated rois
path_annot = '../ANNOTATIONS_INTC41/INCT41/'  # location of bbox annotations
path_audio = '../AUDIO_INTC41/INCT41/'  # location of raw audio
path_save = '../train_dataset/'  # location to save new samples
nombre = next(walk(r"C:\Users\sebas\Documents\GitHub\TESIS-VScode\ANNOTATIONS_INTC41\INCT41"), (None, None, []))[2]  # [] if no file
df = pd.DataFrame()

# _______TRIMMING AUDIO SETTINGS _______
# Set the length of each audio chunk in seconds
chunk_length = 5
# Specify the source and destination directories for the audio files
src_dir = '../AUDIO_INTC41/INCT41/'
dst_dir = '../SCRIPTS/TDL/PHYCUV/AUDIO_TRIM/'

#________PATHS FOR GETTING SPECTROGRAM FILES________
input_folder = '../SCRIPTS/TDL/PHYCUV/AU_PR8/'
output_folder = '../SCRIPTS/TDL/PHYCUV/AUSPEC/'


#_______SETTING UP THE SPECTROGRAM FOLDER TO GENERATE THE DATAFRAME WITH THE INFORMATION_______
spectrogram_folder = '../SCRIPTS/TDL/PHYCUV/AUSPEC/'

#_______SETTING UP THE LABEL DATAFRAME PATH TO GENERATE THE DATAFRAME WITH THE INFORMATION_______
input_folder_LBL = '../SCRIPTS/TDL/PHYCUV/CSV/'

#%%
#________________________BEGINNING OF FUNCTIONS_______________


def filter_window(df, start, end, step):
    """
    Filter a dataframe to get windows of time periods.

    This function filters a dataframe based on start, end and step time values, creating a new dataframe for each
    time window. The resulting dataframes are grouped by the file name and stored in a dictionary.

    Parameters
    ----------
    df (pandas.DataFrame): Dataframe to be filtered.
    start (int): Start time value in seconds.
    end (int): End time value in seconds.
    step (int): Step value in seconds to create new time windows.

    Returns
    -------
    tuple: A tuple containing:
        df_mlabel (list): A list of dataframes containing only rows with time between start and end values.
        fname_lists (dict): A dictionary of dataframes grouped by the file name.
    """
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
    """
       Filter the dataframe row based on specific labels.

    Parameters
    ----------
    row : pd.Series
        The row of the dataframe that needs to be filtered.

    Returns
    -------
    bool
        True if the label in the row is one of the specific labels ('PHYCUV_M', 'PHYCUV_F','BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F'), otherwise False.

    """
    return row['label'] in ['PHYCUV_M', 'PHYCUV_F','BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F']


  
def save_fname_lists(fname_lists, save_path):
    """
        This function saves the data in the `fname_lists` dictionary to csv files. 

    Parameters
    ----------
    fname_lists (dict): A dictionary where the key is a string representing the name of an audio file,
                        and the value is a list of dataframes.
    save_path (str): The path to the directory where the csv files should be saved.

    Returns
    -------
    None

    Example:
    save_fname_lists(fname_lists, "data/csv_files")
    """
    for fname, dfs in fname_lists.items():
        dir_path = os.path.join(save_path, fname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i, df in enumerate(dfs):
            file_path = os.path.join(dir_path, f"{fname.split('.')[0]}_{i}.csv")
            df.to_csv(file_path, index=False)  
 

def trim_audio_files(src_dir, dst_dir, chunk_length):
    """
        Trim the audio files in `src_dir` into smaller chunks of `chunk_length` seconds and save them in `dst_dir`.

    Each audio file in `src_dir` will be divided into chunks of `chunk_length` seconds and saved in a new folder with the same name as the original file in `dst_dir`. Each chunk will have a unique filename consisting of the original filename followed by an underscore and a chunk index.

    Parameters
    ----------
    src_dir : str
        The path to the source directory containing the audio files.
    dst_dir : str
        The path to the destination directory where the trimmed audio chunks will be saved.
    chunk_length : float
        The length of each audio chunk in seconds.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that all audio files in `src_dir` are in WAV format.
    """
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
    

def load_annotations(path_annot, nombre):
    """
        Loads annotations from .txt files in the given directory and returns a dataframe.

    Parameters
    ----------
    path_annot : str
        The path to the folder containing the annotation files.
    nombre : list
        A list of strings representing the names of the annotation files (without the '.txt' extension).

    Returns
    -------
    df_mlabel : pd.DataFrame
        The dataframe containing the loaded annotations.
    fname_lists : list
        A list of filenames corresponding to the loaded annotations.
    """
    df = pd.DataFrame()
    for i in range (len(nombre)):
        flist = glob.glob(path_annot + nombre[i])
        for fname in flist:
            df_aux = util.read_audacity_annot(fname)
            df_aux['fname'] = os.path.basename(fname).replace('.txt', '.wav')
            df_aux = df_aux.drop(columns=['min_f', 'max_f'])
            df = pd.concat([df,df_aux],ignore_index=True)
            df.reset_index(inplace=True, drop=True)
            df_mlabel, fname_lists = filter_window(df,0,60,5)
    return df_mlabel, fname_lists
    
    
def generate_spectrogram(audio_file_path, output_file_path, sr=22050, n_fft=2048, hop_length=512):
    """
    Generate spectrogram from audio file and save it as an image.
    
    Parameters
    ----------
    audio_file_path : str
        Path to audio file
    output_file_path : str
        Path to save the generated spectrogram
    sr : int, optional
        Sampling rate of the audio file, by default 22050
    n_fft : int, optional
        Length of the FFT window, by default 2048
    hop_length : int, optional
        Number of samples between successive frames, by default 512
    
    Returns
    -------
    None
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    log_S = librosa.amplitude_to_db(S)
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
    plt.axis('off')
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()




def generate_spectrograms_from_folder(input_folder, output_folder):
    """
    Generates spectrograms from WAV files in a given folder and saves the spectrograms in a specified output folder.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the WAV files.
    output_folder : str
        The path to the folder where the spectrograms will be saved. If the folder doesn't exist, it will be created.

    Returns
    -------
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            input_file_path = os.path.join(subdir, file)
            if input_file_path.endswith('.wav'):
                output_file_path = os.path.join(output_folder, subdir.replace(input_folder, ''), file[:-4] + '.png')
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                generate_spectrogram(input_file_path, output_file_path)  
                
                
def get_spectrogram_paths(directory):
    """
    Return a list of spectrogram file paths found in the directory.

    Parameters
    ----------
    directory : str
        The directory to search for spectrogram files.

    Returns
    -------
    list
        List of spectrogram file paths.
    """
    spectrogram_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                spectrogram_path = os.path.join(root, file)
                spectrogram_paths.append(spectrogram_path)
    return spectrogram_paths


def get_labels_from_csv(csv_file_path):
    """
    This function reads a csv file and returns a list of labels.

    Parameters
    ----------
    csv_file_path : str
        The path to the csv file that contains the labels.

    Returns
    -------
    list
        A list of strings representing the labels.
    """
    df = pd.read_csv(csv_file_path)
    return df['label'].tolist()


def create_spectrogram_dataframe(directory):
    """
    Creates a Pandas DataFrame containing the names and file paths of spectrograms in a directory.
    
    Parameters
    ----------
    directory : str
        The directory containing the spectrogram files.
    
    Returns
    -------
    pandas.DataFrame
        A dataframe containing two columns: 'NAME', the basename of the spectrogram file without the '.png' extension, 
        and 'Path', the file path to the spectrogram.
    """    
    spectrogram_paths = get_spectrogram_paths(directory)
    spectrogram_names = [os.path.basename(path).replace('.png', '') for path in spectrogram_paths]
    spectrogram_df = pd.DataFrame({'NAME': spectrogram_names, 'Path': spectrogram_paths})
    return spectrogram_df


def create_label_df(input_folder):
    """
    Creates a label DataFrame based on csv files in the input folder.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the csv files.

    Returns
    -------
    label_df : pandas DataFrame
        The created label DataFrame containing the labels of the csv files.

    """
    label_dict = {
        'PHYCUV_M': 0,
        'PHYCUV_F': 0,
        'BOAALB_M': 0,
        'BOAALB_F': 0,
        'BOALUN_F': 0,
        'BOALUN_M': 0,
        'PHYCUV_M': 0,
        'PHYCUV_F': 0
    }
    label_df = pd.DataFrame(columns=['NAME'] + list(label_dict.keys()))
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                labels = get_labels_from_csv(csv_file_path)
                file_name = os.path.splitext(file)[0]
                label_dict = {key: 1 if key in labels else 0 for key in label_dict}
                label_df = label_df.append({'NAME': file_name, **label_dict}, ignore_index=True)
    return label_df


def save_merged_df(merged_df, save_path):
    """
     Saves the dataframes to a csv file.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Dataframe to be saved.
    save_path : str
        The path to save the csv file.

    Returns
    -------
    None
    """
    file_path = os.path.join(save_path, "label_df.csv")
    merged_df.to_csv(file_path, index=False)
                
                  
print("_____________DONE LOADING THE FUNCTIONS_____________")
#__________________________END OF FUNCTIONS_____________________________________


