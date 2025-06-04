import os

# Define the base directory and the outputs folder
base_dir = "./Experiment/PSO/images"
outputs_folder = "./all_files"

# Ensure the outputs folder exists
os.makedirs(outputs_folder, exist_ok=True)

# Define the file to be copied
file_to_copy = "lung_reconstructed_palette_iter_5.png"

# Iterate through all immediate subdirectories in the base directory
for dir_name in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, dir_name)
    
    # Ensure it's a directory
    if os.path.isdir(subfolder_path):
        print(f"Checking in subfolder: {subfolder_path}")

        # Extract the numeric name of the subdirectory
        numeric_name = dir_name
        print(f"Numeric name extracted: {numeric_name}")
        
        # Check if the file exists in the current subfolder
        file_path = os.path.join(subfolder_path, "lung_position_clip/velocity_clip", file_to_copy)
        print(f"Looking for file: {file_path}")
        if os.path.exists(file_path):
            # Define the new file name and destination path
            new_file_name = f"{numeric_name}.png"
            print(f"New file name: {new_file_name}")
            destination_path = os.path.join(outputs_folder, new_file_name)
            print(f"Destination path: {destination_path}")
            
            # Copy and rename the file using os.system
            os.system(f"cp '{file_path}' '{destination_path}'")
            print(f"File '{file_to_copy}' has been copied and renamed to '{destination_path}'.")
        else:
            print(f"File '{file_to_copy}' does not exist in '{subfolder_path}'.")
            print(f"Trying for lung_reconstructed_palette_iter_0.png")
            # Try the alternative file name
            alternative_file_path = os.path.join(subfolder_path, "lung_position_clip/velocity_clip", "lung_reconstructed_palette_iter_0.png")
            if os.path.exists(alternative_file_path):
                new_file_name = f"{numeric_name}_iter_0.png"
                destination_path = os.path.join(outputs_folder, new_file_name)
                os.system(f"cp '{alternative_file_path}' '{destination_path}'")
                print(f"File 'lung_reconstructed_palette_iter_0.png' has been copied and renamed to '{destination_path}'.")
            else:
                print(f"Neither '{file_to_copy}' nor 'lung_reconstructed_palette_iter_0.png' exists in '{subfolder_path}'.")
