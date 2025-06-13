import os

"""
In order to perform PSO, it is necessary to have all the files in 1 directory.
If files are divided into train, val, test_1, test_2, and test_3, this script 
will collect all files from these directories and copy them into a single directory.
"""

def collect_all_files():
    # Define the root directory (current directory) and the target directory
    root_dir = os.getcwd() 
    target_dir = os.path.join(root_dir, 'all_files')

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    subdirs = ['train', 'val', 'test', 'test_1', 'test_2', 'test_3']
    image_extensions = ('.png', '.jpg', '.jpeg')

    # Initialize counters
    file_count = 0
    total_files = 0

    # Iterate through each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.exists(subdir_path):  # Check if the subdirectory exists
            for root, _, files in os.walk(subdir_path):
                total_files += len(files)  # Count all files in the subdirectory
                for file in files:
                    if file.lower().endswith(image_extensions):  # Check if the file is an image
                        source_path = os.path.join(root, file)
                        destination_path = os.path.join(target_dir, file)

                        # Avoid overwriting files with the same name
                        if os.path.exists(destination_path):
                            print(f"Skipping file '{file}' as it already exists in the target directory.")
                            continue

                        # Copy the file
                        with open(source_path, 'rb') as src_file:
                            with open(destination_path, 'wb') as dest_file:
                                dest_file.write(src_file.read())
                        file_count += 1

    print(f"Total files in subdirectories: {total_files}")
    print(f"Collected {file_count} image files into the 'all_files' directory.")


if __name__ == "__main__":
    collect_all_files()
