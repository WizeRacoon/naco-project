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

    # Filter images for PA and AP views and specific labels
    def filter_images(image_list):
        filtered = []
        for image_name in image_list:
            if image_name in labels and image_name in all_image_paths:
                label_info = labels[image_name]
                if label_info['view_position'] in ['PA', 'AP'] and (
                    'Atelectasis' in label_info['labels'].split('|') or label_info['labels'] == 'No Finding'
                ):
                    filtered.append({
                        'image_name': image_name,
                        'path': all_image_paths[image_name],
                        'labels': label_info['labels'],
                        'view_position': label_info['view_position']
                    })
        return filtered

    # Filter test and train/val images
    test_data = filter_images(test_images)
    train_val_data = filter_images(train_val_images)

    # Balance the dataset for PA and AP views
    def balance_data(data, total_count):
        atelectasis_pa = [img for img in data if 'Atelectasis' in img['labels'].split('|') and img['view_position'] == 'PA']
        atelectasis_ap = [img for img in data if 'Atelectasis' in img['labels'].split('|') and img['view_position'] == 'AP']
        normal_pa = [img for img in data if img['labels'] == 'No Finding' and img['view_position'] == 'PA']
        normal_ap = [img for img in data if img['labels'] == 'No Finding' and img['view_position'] == 'AP']

        # Ensure equal number of PA and AP files
        max_per_view = total_count // 4  # Divide equally among PA/AP for both labels
        atelectasis_pa = atelectasis_pa[:max_per_view]
        atelectasis_ap = atelectasis_ap[:max_per_view]
        normal_pa = normal_pa[:max_per_view]
        normal_ap = normal_ap[:max_per_view]

        # Combine and shuffle
        balanced_data = atelectasis_pa + atelectasis_ap + normal_pa + normal_ap
        random.shuffle(balanced_data)
        return balanced_data

    # Split train, val, and test datasets
    train_data = balance_data(train_val_data, 6986)
    val_data = balance_data(train_val_data, 1748)
    test_data = balance_data(test_data, 2722)

    # Split test into three balanced subsets
    def split_test_data(data, num_splits):
        split_size = len(data) // num_splits
        return [data[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    test_splits = split_test_data(test_data, 3)

    # Function to save images into subdirectories
    def save_images_to_subdirs(images, split_name):
        split_dir = os.path.join('./PA-AP_atelectasis-normal', split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        normal_dir = os.path.join(split_dir, 'NORMAL')
        atelectasis_dir = os.path.join(split_dir, 'ATELECTASIS')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(atelectasis_dir, exist_ok=True)
        
        for img in images:
            source_path = img['path']
            if img['labels'] == 'No Finding':
                destination_path = os.path.join(normal_dir, os.path.basename(source_path))
            elif 'Atelectasis' in img['labels'].split('|'):
                destination_path = os.path.join(atelectasis_dir, os.path.basename(source_path))
            else:
                continue  # Skip images that don't match the desired labels
            
            # Copy file manually
            with open(source_path, 'rb') as src_file:
                with open(destination_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())

    # Save images to train, test, and val directories
    save_images_to_subdirs(train_data, 'train')
    save_images_to_subdirs(val_data, 'val')
    save_images_to_subdirs(test_data, 'test')

    for i, split in enumerate(test_splits):
        save_images_to_subdirs(split, f'test_{i + 1}')

    print(f"Saved {len(train_data)} images to train directory.")
    print(f"Saved {len(val_data)} images to val directory.")
    print(f"Saved {len(test_data)} images to test directory.")
    for i, split in enumerate(test_splits):
        print(f"Saved {len(split)} images to test_{i + 1} directory.")

main()