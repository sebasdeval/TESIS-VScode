#%%
import pandas as pd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
#%%
# Read the dataframe
df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_3_Labels.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataframe


# Determine the label with the least amount of information
label_counts = df.drop(['NAME', 'Path'], axis=1).sum()
label_with_less_info = label_counts.idxmin()

# Extract the paths of the spectrogram images
image_paths = df['Path'].tolist()

# Initialize an empty dataframe to store the augmented data
augmented_data = pd.DataFrame(columns=df.columns)

# Initialize the directory for the augmented images
aug_dir = '../SCRIPTS/TDL/PHYCUV/AUG/'
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

# Loop through each image path
for path in image_paths:
    # Load the spectrogram image
    img = plt.imread(path)
    
    # Apply time mask
    n_steps = 4
    img_height, img_width, num_channels = img.shape
    step_size = img_width // n_steps
    for i in range(n_steps):
        mask = np.ones((img_height, img_width, num_channels), dtype=np.float32)
        start = i * step_size
        end = min(start + step_size, img_width)
        mask[:, start:end, :] = 0  # apply mask to all channels
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







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataframe
df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_3_Labels.csv')

# Determine the label with the least amount of information
label_counts = df.drop(['NAME', 'Path'], axis=1).sum()
label_with_less_info = label_counts.idxmin()

# Extract the paths of the spectrogram images
image_paths = df['Path'].tolist()

# Initialize an empty list to store the augmented data
augmented_data = []

# Initialize the directory for the augmented images
aug_dir = '../SCRIPTS/TDL/PHYCUV/AUG/'
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

# Calculate the maximum number of samples per label
max_samples = df.drop(['NAME', 'Path'], axis=1).sum().max()

# Loop through each label
for label in df.drop(['NAME', 'Path'], axis=1).columns:
    # Extract the paths of the spectrogram images with the current label
    label_image_paths = df[df[label] == 1]['Path'].tolist()
    
    # Calculate the number of samples to be augmented for the current label
    num_samples = max_samples - len(label_image_paths)
    
    # Loop through each image path with the current label
    for path in label_image_paths:
        # Load the spectrogram image
        img = plt.imread(path)
        
        # Apply time mask
        n_steps = 4
        img_height, img_width, num_channels = img.shape
        step_size = img_width // n_steps
        for i in range(n_steps):
            mask = np.ones((img_height, img_width, num_channels), dtype=np.float32)
            start = i * step_size
            end = min(start + step_size, img_width)
            mask[:, start:end, :] = 0  # apply mask to all channels
            masked_img = img * mask
            
            # Save the augmented image with a unique name
            filename, ext = os.path.splitext(os.path.basename(path))
            new_filename = f'{filename}_AUG_{i}{ext}'
            new_path = os.path.join(aug_dir, new_filename)
            plt.imsave(new_path, masked_img)
            
            # Add the augmented data to the list
            new_row = df[df['Path'] == path].copy()
            new_row['Path'] = new_path
            new_row[label_with_less_info] = 1
            augmented_data.append(new_row)
    
    # Loop through each image path with the current label and add the original data to the list
    for path in label_image_paths:
        new_row = df[df['Path'] == path].copy()
        augmented_data.append(new_row)
    
    # Loop through each augmented sample for the current label and add it to the list
    label_augmented_data = []
    for i in range(num_samples):
        # Select a random image path with the current label
        path = np.random.choice(label_image_paths)
        
        # Load the spectrogram image
        img = plt.imread(path)
        
        # Apply time mask
        n_steps = 4
        img_height, img_width, num_channels = img.shape
        step_size = img_width // n_steps
        for i in range(n_steps):
            mask = np.ones((img_height, img_width, num_channels), dtype=np.float32)
            start

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_COMPLETE.csv')

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
aug_dir = '../SCRIPTS/TDL/PHYCUV/AUG/'
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

# Loop through each image path
for path in image_paths:
    # Load the spectrogram image
    img = plt.imread(path)

    # If the sample belongs to the minority class, augment it
    if df.loc[df['Path'] == path, label_with_less_info].values[0] == 1:
        # Apply time mask
        n_steps = 90
        img_height, img_width, num_channels = img.shape
        step_size = img_width // n_steps
        for i in range(n_steps):
            mask = np.ones((img_height, img_width, num_channels), dtype=np.float32)
            start = i * step_size
            end = min(start + step_size, img_width)
            mask[:, start:end, :] = 0  # apply mask to all channels
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
#%%
#df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_3_Labels.csv')
# Get the count of each label in the balanced dataset
label_counts = balanced_data.drop(['NAME', 'Path'], axis=1).sum()

# Print the count of each label
print(label_counts)

# %%
