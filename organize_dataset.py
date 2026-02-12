import os
import shutil
import pandas as pd

# Paths
dataset_path = "train"
csv_path = "labels.csv"
output_path = "dataset/train"

# Read CSV
labels = pd.read_csv(csv_path)

# Create output folder
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through each row
for index, row in labels.iterrows():
    breed = row['breed']
    image_id = row['id'] + ".jpg"

    breed_folder = os.path.join(output_path, breed)

    if not os.path.exists(breed_folder):
        os.makedirs(breed_folder)

    src_path = os.path.join(dataset_path, image_id)
    dst_path = os.path.join(breed_folder, image_id)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

print("Dataset Organized Successfully!")