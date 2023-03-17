#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the dataset
df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/Datos_70_Entrenamiento_para_augTRAIN.csv')

# Get the count of each label in the dataset
label_counts = df.drop(['NAME', 'Path'], axis=1).sum()

# Determine the label with the least amount of information
label_with_less_info = label_counts.idxmin()

# Get the number of samples with a value of 1 for the least common label
n_samples_to_add = label_counts.max() - label_counts.min()

# Extract the paths of the spectrogram images
image_paths = df['Path'].tolist()

# Initialize an empty dataframe to store the augmented data
augmented_data = pd.DataFrame(columns=df.columns)

# Initialize the directory for the augmented images
aug_dir = '../SCRIPTS/TDL/PHYCUV/AUG_FREQUENCY_MASKING/'
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

# Loop through each image path
for path in image_paths:
    # Load the spectrogram image
    img = plt.imread(path)

    # If the sample belongs to the minority class, augment it
    if df.loc[df['Path'] == path, label_with_less_info].values[0] == 1:
        # Apply frequency mask
        n_steps = 15
        img_height, img_width, num_channels = img.shape
        step_size = img_height // n_steps
        for i in range(n_steps):
            mask = np.ones((img_height, img_width, num_channels), dtype=np.float32)
            start = i * step_size
            end = min(start + step_size, img_height)
            mask[start:end, :, :] = 0  # apply mask to all channels
            masked_img = img * mask

            # Save the augmented image with a unique name
            filename, ext = os.path.splitext(os.path.basename(path))
            new_filename = f'{filename}_AUG_{i}{ext}'
            new_path = os.path.join(aug_dir, new_filename)
            plt.imsave(new_path, masked_img)

            # Add the augmented data to the dataframe
            new_row = df[df['Path'] == path].copy()
            new_row['Path'] = new_path
            new_row[label_with_less_info] = 1
            augmented_data = augmented_data.append(new_row)

# Concatenate the original dataframe and the augmented dataframe
balanced_data = pd.concat([df, augmented_data])

# Get the count of each label in the balanced dataset
label_counts = balanced_data.drop(['NAME', 'Path'], axis=1).sum()

# Print the count of each label
print(label_counts)
# %%
balanced_data.to_csv('../SCRIPTS/TDL/PHYCUV/DATASET/Datos_70_PLUS_AUG_FRECMSK.csv', index=False)
# %%

# %%
