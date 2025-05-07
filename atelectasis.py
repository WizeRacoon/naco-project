# This is the main file of the project

import data_preprocessing as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df
import os
import random
from math import floor

# Preprocessing data
csv_file_path = 'input_data/labels.csv'
images_directory = 'input_data/images'

# Create image objects
image_data_objects = dp.create_image_objects(csv_file_path, images_directory)

# Filter image objects for labels 'Atelectasis' and 'No Finding'
filtered_image_data_objects = [
    obj for obj in image_data_objects if 'Atelectasis' in obj.labels.split('|') or obj.labels == 'No Finding'
]

# Separate filtered image objects into two groups
atelectasis_objects = [
    obj for obj in filtered_image_data_objects if 'Atelectasis' in obj.labels.split('|')
]
no_finding_objects = [
    obj for obj in filtered_image_data_objects if obj.labels == 'No Finding'
]

# Determine the maximum number of objects to take from each group 
max_objects_per_label = min(len(atelectasis_objects), len(no_finding_objects))

# Take an equal number of objects from each group
balanced_image_data_objects = atelectasis_objects[:max_objects_per_label] + no_finding_objects[:max_objects_per_label]

# Set seed for reproducibility
random.seed(42)
# Shuffle the balanced dataset
random.shuffle(balanced_image_data_objects)

# Update the total_images variable to reflect the new dataset size
total_images = len(balanced_image_data_objects)

# Calculate split sizes
train_size = floor(0.6 * total_images)
test_size = floor(0.3 * total_images)
val_size = total_images - train_size - test_size  # Remaining images go to val

# Split the data
train_images = balanced_image_data_objects[:train_size]
test_images = balanced_image_data_objects[train_size:train_size + test_size]
val_images = balanced_image_data_objects[train_size + test_size:]

# Function to save images into subdirectories
def save_images_to_subdirs(images, split_name):
    split_dir = os.path.join('input_data/filtered_images_split', split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    normal_dir = os.path.join(split_dir, 'NORMAL')
    atelectasis_dir = os.path.join(split_dir, 'ATELECTASIS')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(atelectasis_dir, exist_ok=True)
    
    for obj in images:
        source_path = obj.image_path  # Path to the original image
        if obj.labels == 'No Finding':
            destination_path = os.path.join(normal_dir, os.path.basename(source_path))
        elif 'Atelectasis' in obj.labels.split('|'):
            destination_path = os.path.join(atelectasis_dir, os.path.basename(source_path))
        else:
            continue  # Skip images that don't match the desired labels
        
        # Copy file manually
        with open(source_path, 'rb') as src_file:
            with open(destination_path, 'wb') as dest_file:
                dest_file.write(src_file.read())

# Save images to train, test, and val directories
save_images_to_subdirs(train_images, 'train')
save_images_to_subdirs(test_images, 'test')
save_images_to_subdirs(val_images, 'val')

print(f"Saved {len(train_images)} images to train directory.")
print(f"Saved {len(test_images)} images to test directory.")
print(f"Saved {len(val_images)} images to val directory.")


''' MIKA ATELECTASIS
# This is the main file of the project
# Import subscripts 
import data_processor as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df

csv_file_path = 'data/input/labels.csv'
images_directory = 'data/input/images'
    

def main():
    image_data_objects = dp.create_image_objects(csv_file_path, images_directory)

    print(f"Created {len(image_data_objects)} image data objects.")
    print('\n')

    #image_data_objects[0].print_details() # example

    for image_data_object in image_data_objects:
        img_name = image_data_object.image_index.split(".")[0]

        nPSO.PSO(n=20, N=3, image_path= f"{images_directory}/{image_data_object.image_index}", max_iterations=10)
        pPSO.lung_mask(image_index = image_data_object.image_index, 
                       PSO_image_relative_path = f"data/PSO/images/{img_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_9.png", 
                       output_directory = "data/masked_lungs/images")
        image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(f"data/masked_lungs/images/{image_data_object.image_index}", show_symmetry_line=False)
        
        image_data_object.print_details()

    df.calc_lung_symmetry(f"OUTPUT_LUNG_RESULTS/lung/lung_position_clip/velocity_clip/lung_post_processed_iter_9.png", show_symmetry_line=True)
    df.calc_lung_symmetry(f"OUTPUT_LUNG_RESULTS/lung2/lung_position_clip/velocity_clip/lung_post_processed_iter_9.png", show_symmetry_line=True)

if __name__ == "__main__":
    main()  # call the main function when the script is run directly
'''