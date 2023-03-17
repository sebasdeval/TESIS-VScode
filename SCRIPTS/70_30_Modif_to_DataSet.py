#%%
import pandas as pd
import numpy as np

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_COMPLETE_3_Labels.csv')

# Separate the data based on the label columns
groups = df.groupby(['PHYCUV', 'BOAALB', 'BOALUN'])

# Define the percentages for each resulting dataframe
split_perc = 0.35

# Initialize empty lists for the resulting dataframes
df_30 = []
df_70 = []

# Loop through each group and split into 30% and 70% dataframes
for name, group in groups:
    # Randomly shuffle the rows
    group = group.sample(frac=1, random_state=24)

    # Split into 30% and 70% dataframes
    split_index = int(len(group) * split_perc)
    group_30 = group.iloc[:split_index]
    group_70 = group.iloc[split_index:]

    # Append to the resulting dataframes lists
    df_30.append(group_30)
    df_70.append(group_70)

# Concatenate the resulting dataframes
df_30 = pd.concat(df_30)
df_70 = pd.concat(df_70)

# Drop any duplicate rows in each resulting dataframe
df_30 = df_30.drop_duplicates()
df_70 = df_70.drop_duplicates()
#%%
train_label_counts = np.sum(df_30, axis=0)
test_label_counts = np.sum(df_70, axis=0)
print(f"Train label counts: {train_label_counts}")
print(f"Test label counts: {test_label_counts}")
# %%
#Comprobation
treinta= df_30['Path'].tolist()
setenta = df_70['Path'].tolist()
if set(setenta) & set(treinta):
    print("There are common elements between the two lists.")
else:
    print("There are no common elements between the two lists.")
    
#%%
#Saving dataframe
df_30.to_csv('../SCRIPTS/TDL/PHYCUV/DATASET/Datos_30_comprobacion_para_augVAL.csv', index=False) 
df_70.to_csv('../SCRIPTS/TDL/PHYCUV/DATASET/Datos_70_Entrenamiento_para_augTRAIN.csv', index=False)

# %%
df1 = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_JUST_AUGMENTED_3_Labels.csv',delimiter=';')

# %%
df1 = df1.append(df_70, ignore_index=True)
# %%
train_label_counts = np.sum(df1, axis=0)

print(f"Train label counts: {train_label_counts}")

# %%
#Comprobation
df1['Path'].value_counts()
#%%
#Saving Df intended to be used in augmented models training 
df1.to_csv('../SCRIPTS/TDL/PHYCUV/DATASET/Datos_70_Entrenamiento_plus_augdata.csv', index=False)
# %%
