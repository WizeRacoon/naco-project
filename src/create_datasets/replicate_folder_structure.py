import os

def main():
    # File paths
    raw_directory = './only_PA-and-A'  # Source directory with the desired structure
    images_root_directory = './results_ONLY-PA-ATELECTASIS/all_files'  # Root directory containing images in subfolders
    target_directory = './results_ONLY-PA-ATELECTASIS_pso5'  # Target directory to replicate structure

    # Recursively find all image files in the subdirectories
    def find_all_images(root_dir):
        image_files = {}
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files[file] = os.path.join(subdir, file)
        return image_files

    # Copy files to the target directory while maintaining the structure
    def replicate_structure(raw_dir, images_map, target_dir):
        os.makedirs(target_directory, exist_ok=True)
        for subdir, _, files in os.walk(raw_dir):
            relative_subdir = os.path.relpath(subdir, raw_dir)
            target_subdir = os.path.join(target_dir, relative_subdir)
            os.makedirs(target_subdir, exist_ok=True)

            for file in files:
                if file in images_map:
                    source_path = images_map[file]
                    destination_path = os.path.join(target_subdir, file)
                    
                    # Copy file manually
                    with open(source_path, 'rb') as src_file:
                        with open(destination_path, 'wb') as dest_file:
                            dest_file.write(src_file.read())
                else:
                    print(f"Warning: File '{file}' not found in all_pso_images.")

    # Find all images in all_pso_images
    all_image_paths = find_all_images(images_root_directory)

    # Replicate the structure and copy files
    replicate_structure(raw_directory, all_image_paths, target_directory)

    print(f"Files copied to '{target_directory}' maintaining the structure of '{raw_directory}'.")

main()