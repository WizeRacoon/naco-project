import data_processor as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df
import os
import random
from math import floor

def main():
    # Preprocessing data
    csv_file_path = 'data/input/labels.csv'
    images_directory = 'data/input/images'

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

    print("here?")

    # Function to save images into subdirectories
    def save_images_to_subdirs(images, split_name):
        split_dir = os.path.join('data/input/filtered_images_split', split_name)
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
    
    #image_data_objects = dp.create_image_objects(csv_file_path, images_directory)
    #print(f"Created {len(image_data_objects)} image data objects.")
    #print('\n')
    #image_data_objects[0].print_details() # example

    for image_data_object in atelectasis_objects:
        img_name = image_data_object.image_index.split(".")[0]

        subfolder_name = ""
        images_test = os.listdir("data/input/filtered_images_split/test/ATELECTASIS")
        images_train = os.listdir("data/input/filtered_images_split/train/ATELECTASIS")
        images_val = os.listdir("data/input/filtered_images_split/val/ATELECTASIS")

        for image in images_test:
            if image == image_data_object.image_index:
                subfolder_name = "test"
        for image in images_train:
            if image == image_data_object.image_index:
                subfolder_name = "train"
        for image in images_val:
            if image == image_data_object.image_index:
                subfolder_name = "val"

        nPSO.PSO(n=20, N=3, image_path= f"{images_directory}/{image_data_object.image_index}", max_iterations=10)
        pPSO.lung_mask(image_index = image_data_object.image_index, 
                       PSO_image_relative_path = f"data/PSO/images/{img_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_9.png", 
                       output_directory = f"data/input/filtered_images_split/{subfolder_name}/ATELECTASIS")
        #image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(f"data/masked_lungs/images/{image_data_object.image_index}", show_symmetry_line=False)
        
        #image_data_object.print_details()
    
    for image_data_object in no_finding_objects:
        img_name = image_data_object.image_index.split(".")[0]

        subfolder_name = ""
        images_test = os.listdir("data/input/filtered_images_split/test/NORMAL")
        images_train = os.listdir("data/input/filtered_images_split/train/NORMAL")
        images_val = os.listdir("data/input/filtered_images_split/val/NORMAL")

        for image in images_test:
            if image == image_data_object.image_index:
                subfolder_name = "test"
        for image in images_train:
            if image == image_data_object.image_index:
                subfolder_name = "train"
        for image in images_val:
            if image == image_data_object.image_index:
                subfolder_name = "val"

        nPSO.PSO(n=20, N=3, image_path= f"{images_directory}/{image_data_object.image_index}", max_iterations=10)
        pPSO.lung_mask(image_index = image_data_object.image_index, 
                       PSO_image_relative_path = f"data/PSO/images/{img_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_9.png", 
                       output_directory = f"data/input/filtered_images_split/{subfolder_name}/NORMAL")