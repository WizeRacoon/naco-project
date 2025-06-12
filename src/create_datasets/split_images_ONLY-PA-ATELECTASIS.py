import os
import random
from math import floor
import time

def main():
    # File paths
    csv_file_path = 'archive/Data_Entry_2017.csv'
    test_list_path = 'archive/test_list.txt'
    train_val_list_path = 'archive/train_val_list.txt'
    images_root_directory = 'archive'  # Root directory containing images in subfolders

    # Load labels from Data_Entry_2017.csv
    def load_labels(csv_path):
        labels = {}
        with open(csv_path, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(',')
                image_name = parts[0]
                finding_labels = parts[1]
                view_position = parts[6]
                labels[image_name] = {'labels': finding_labels, 'view_position': view_position}
        return labels

    # Load image lists from test_list.txt and train_val_list.txt
    def load_image_list(file_path):
        with open(file_path, 'r') as file:
            return set(line.strip() for line in file)

    # Recursively find all image files in the subdirectories
    def find_all_images(root_dir):
        image_files = {}
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files[file] = os.path.join(subdir, file)
        return image_files

    # Load data
    labels = load_labels(csv_file_path)
    test_images = load_image_list(test_list_path)
    train_val_images = load_image_list(train_val_list_path)
    all_image_paths = find_all_images(images_root_directory)

    # Filter images for PA view and specific labels
    def filter_images(image_list):
        filtered = []
        for image_name in image_list:
            if image_name in labels and image_name in all_image_paths:
                label_info = labels[image_name]
                if label_info['view_position'] == 'PA' and (
                    label_info['labels'] == 'Atelectasis' or label_info['labels'] == 'No Finding'
                ):
                    filtered.append({
                        'image_name': image_name,
                        'path': all_image_paths[image_name],
                        'labels': label_info['labels']
                    })
        return filtered

    # Filter test and train/val images
    test_data = filter_images(test_images)
    train_val_data = filter_images(train_val_images)

    # Balance the test dataset
    def balance_test_data(data):
        atelectasis = [img for img in data if img['labels'] == 'Atelectasis']
        no_finding = [img for img in data if img['labels'] == 'No Finding']
        max_objects_per_label = min(len(atelectasis), len(no_finding))
        balanced_data = atelectasis[:max_objects_per_label] + no_finding[:max_objects_per_label]
        random.shuffle(balanced_data)
        return balanced_data

    test_data = balance_test_data(test_data)

    # Separate train/val data into ATELECTASIS and NORMAL
    atelectasis = [img for img in train_val_data if img['labels'] == 'Atelectasis']
    no_finding = [img for img in train_val_data if img['labels'] == 'No Finding']

    # Balance the dataset for train/val
    max_objects_per_label = min(len(atelectasis), len(no_finding))
    atelectasis = atelectasis[:max_objects_per_label]
    no_finding = no_finding[:max_objects_per_label]

    # Shuffle the data
    random.seed(42)
    random.shuffle(atelectasis)
    random.shuffle(no_finding)

    # Split each label into train and validation
    train_size_atelectasis = floor(0.8 * len(atelectasis))
    train_size_no_finding = floor(0.8 * len(no_finding))

    train_atelectasis = atelectasis[:train_size_atelectasis]
    val_atelectasis = atelectasis[train_size_atelectasis:]

    train_no_finding = no_finding[:train_size_no_finding]
    val_no_finding = no_finding[train_size_no_finding:]

    # Combine the splits
    train_data = train_atelectasis + train_no_finding
    val_data = val_atelectasis + val_no_finding

    # Shuffle the combined datasets
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Function to save images into subdirectories
    def save_images_to_subdirs(images, split_name):
        split_dir = os.path.join('./only_PA-and-A', split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        normal_dir = os.path.join(split_dir, 'NORMAL')
        atelectasis_dir = os.path.join(split_dir, 'ATELECTASIS')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(atelectasis_dir, exist_ok=True)
        
        for img in images:
            source_path = img['path']
            if img['labels'] == 'No Finding':
                destination_path = os.path.join(normal_dir, os.path.basename(source_path))
            elif img['labels'] == 'Atelectasis':
                destination_path = os.path.join(atelectasis_dir, os.path.basename(source_path))
            else:
                continue  # Skip images that don't match the desired labels
            
            # Copy file manually
            with open(source_path, 'rb') as src_file:
                with open(destination_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())

    # Save images to train, test, and val directories
    save_images_to_subdirs(train_data, 'train')
    save_images_to_subdirs(test_data, 'test')
    save_images_to_subdirs(val_data, 'val')

    print(f"Saved {len(train_data)} images to train directory.")
    print(f"Saved {len(test_data)} images to test directory.")
    print(f"Saved {len(val_data)} images to val directory.")


def divide_test_directory():
    # Paths
    test_dir = './only_PA-and-A/test'
    output_dirs = ['./only_PA-and-A/test_1', './only_PA-and-A/test_2', './only_PA-and-A/test_3']

    # Subdirectories for NORMAL and ATELECTASIS
    normal_dir = os.path.join(test_dir, 'NORMAL')
    atelectasis_dir = os.path.join(test_dir, 'ATELECTASIS')

    # Get all files in NORMAL and ATELECTASIS
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if os.path.isfile(os.path.join(normal_dir, f))]
    atelectasis_files = [os.path.join(atelectasis_dir, f) for f in os.listdir(atelectasis_dir) if os.path.isfile(os.path.join(atelectasis_dir, f))]

    # Ensure equal number of NORMAL and ATELECTASIS files
    max_objects_per_label = min(len(normal_files), len(atelectasis_files))
    normal_files = normal_files[:max_objects_per_label]
    atelectasis_files = atelectasis_files[:max_objects_per_label]

    # Shuffle the files
    random.seed(42)
    random.shuffle(normal_files)
    random.shuffle(atelectasis_files)

    # Split files into three balanced groups
    split_size = max_objects_per_label // 3
    normal_splits = [normal_files[i * split_size:(i + 1) * split_size] for i in range(3)]
    atelectasis_splits = [atelectasis_files[i * split_size:(i + 1) * split_size] for i in range(3)]

    # Ensure the output directories exist
    for output_dir in output_dirs:
        os.makedirs(os.path.join(output_dir, 'NORMAL'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ATELECTASIS'), exist_ok=True)

    # Copy files to the new test directories
    for i, output_dir in enumerate(output_dirs):
        for file in normal_splits[i]:
            destination_path = os.path.join(output_dir, 'NORMAL', os.path.basename(file))
            with open(file, 'rb') as src_file:
                with open(destination_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
        for file in atelectasis_splits[i]:
            destination_path = os.path.join(output_dir, 'ATELECTASIS', os.path.basename(file))
            with open(file, 'rb') as src_file:
                with open(destination_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())

    print(f"Divided test directory into {len(output_dirs)} balanced subdirectories.")


main()
time.sleep(5)
divide_test_directory()
