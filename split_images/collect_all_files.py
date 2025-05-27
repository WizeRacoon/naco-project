import os

def collect_all_files():
    # Define the root directory (current directory) and the target directory
    root_dir = os.getcwd()  # Current working directory (only_PA folder)
    target_dir = os.path.join(root_dir, 'all_files')

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Subdirectories to search for image files
    subdirs = ['train', 'val', 'test', 'test_1', 'test_2', 'test_3']

    # Supported image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg')

    # Counter for the number of files copied
    file_count = 0

    # Iterate through each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.exists(subdir_path):  # Check if the subdirectory exists
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith(image_extensions):  # Check if the file is an image
                        source_path = os.path.join(root, file)
                        destination_path = os.path.join(target_dir, file)

                        # Avoid overwriting files with the same name
                        if os.path.exists(destination_path):
                            base, ext = os.path.splitext(file)
                            counter = 1
                            while os.path.exists(destination_path):
                                destination_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                                counter += 1

                        # Copy the file
                        with open(source_path, 'rb') as src_file:
                            with open(destination_path, 'wb') as dest_file:
                                dest_file.write(src_file.read())

                        file_count += 1

    print(f"Collected {file_count} image files into the 'all_files' directory.")

# Run the function
if __name__ == "__main__":
    collect_all_files()