import os
import shutil

# Path where 120-breed dataset was created
source_path = "dataset/train"

# New folder for only 20 breeds
target_path = "dataset_20/train"

selected_breeds = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]

# Create new dataset folder
if not os.path.exists(target_path):
    os.makedirs(target_path)

for breed in selected_breeds:
    src = os.path.join(source_path, breed)
    dst = os.path.join(target_path, breed)

    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"{breed} copied successfully!")

print("20 Breeds Dataset Created Successfully!")