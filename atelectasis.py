# This is the main file of the project

import data_processor as dp, PSO as pso, segmentation as seg, differential_function as df
import os
from multiprocessing import Pool, cpu_count
import random
from collections import defaultdict

# Data Filters ==================================================================================
# Here, you can adjust which data to consider. If you want to consider all data of a class, just state the list ["any"] (this cannot be done for the "labels" class)

image_index                     = ["any"]                                   # list of indeces of image; unique identifier in the form "'patient_id' + '_' + 'follow_up_number' + '.png'"
labels                          = ["No Finding", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]             # list of labels (diseases). Possibilities: ["No Finding", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
follow_up_number                = ["any"]                                   # list of number of patients' xray image (e.g, their 25th xray image)
patient_id                      = ["any"]
patient_age                     = ["any"]                                   # list of ages in format 'XXXY', where the 3 X's denote the age, and 'Y' is an abbreviation for year. e.g., '059Y' means the patient is 59 years old
patient_gender                  = ["any"]                                   # list of genders in format ["M","F"]
view_position                   = ["any"]                                   # list of view positions in format ["PA","AP"]. PA means posteroanterior (image trough the front), AP means anteroposterior (image trough the back)
#OriginalImageWidth             =                                           # trivial as the dataset has converted all images to 1024
#OriginalImageHeight            =                                           # trivial as the dataset has converted all images to 1024
#OriginalImagePixelSpacing_x    =                                           # trivial
#OriginalImagePixelSpacing_y    =                                           # trivial

exclusive_label                 = False                                     # if 'True': set such that patient only has one label in the list 'labels' | if 'False': set such that a patient can have multiple labels specified

#================================================================================================

# Experiment Parameters =========================================================================
# Here, you can adjust what experiment you want to run and where to put it

### Experiment name------------------
experiment_name                     = "Experiment_all_N3_i20"

### Multiprocessing------------------
multiprocessing                     = True                                                              # apply multiprocessing?
core_percentage                     = 75                                                                # percentage of CPU to use when multiprocessing

### Input data location--------------
csv_file_path                       = 'data/input/labels.csv'
images_directory                    = 'data/input/images'

### max amount of images-------------
max_images                          = 1000000                                                           # for if you only want to run a few images

### PSO------------------------------
apply_pso                           = False

n                                   = 20                                                                # number of palettes (particles)
N                                   = 3                                                                 # number of colors in each palette
max_iterations                      = 20                                                                # maximum number of iterations
#image_path                         =
pso_output_directory                = f"{experiment_name}/PSO/images"
save_iteration_list                 = [1,2,3,4,5,10,20]                                                 # list of PSO iterations to be saved as an image


### differential function------------
apply_differential_function         = True
on_pso_iteration                    = 20                                                                # iteration at which the differential function is applied

#image_index                        =
#pso_image_relative_path            =
save_symmetry_line                  = True                                                              # save the symmetry line image in differential_output_directory   
save_intermediate_steps             = True                                                              # save the intermedate segmentation steps also (including symmetry line) in differential_output_directory                         
differential_output_directory       = f"{experiment_name}/differential_function/images"
enable_differential_optimization    = True                                                              # at the end apply differential optimization to the best combination of symmetry_percentage and proportional_lung_capacity
number_of_trails                    = 10                                                                # number of times to run the differential optimization (with different random train/test plits)
max_dataset_size                    = 100                                                               # max amount of images to use for differential optimization

#================================================================================================


# Making dirs if they don't exist yet
if apply_pso:
    os.makedirs(pso_output_directory, exist_ok=True)
if save_symmetry_line or save_intermediate_steps:
    os.makedirs(differential_output_directory, exist_ok=True)

def process_image(image_data_object):
    image_name = image_data_object.image_index.split(".")[0]

    if apply_pso:
        print(f"[{image_name}] - applying PSO")
        for save_iteration in save_iteration_list:
            if save_iteration < 1 or save_iteration > max_iterations:
                raise ValueError(f"save_iteration_list value {save_iteration} is invalid.")
        pso.PSO(
            n=n,
            N=N,
            image_path=f"{images_directory}/{image_data_object.image_index}",
            max_iterations=max_iterations,
            output_directory=pso_output_directory,
            save_iteration_list=save_iteration_list
        )
        
    # For if the pso converges earlier and a later iteration does not exist
    for i in range(on_pso_iteration):
        if os.path.isfile(f"{pso_output_directory}/{image_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_{on_pso_iteration - i}.png"):
            pso_iteration = on_pso_iteration - i
        i += 1

    if apply_differential_function:
        print(f"[{image_name}] - applying differential function")
        symmetry_x, segmented_image = df.further_segmentation(
            image_name=image_name,
            pso_image_relative_path=f"{pso_output_directory}/{image_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_{pso_iteration}.png",
            save_symmetry_line=save_symmetry_line,
            save_intermediate_steps=save_intermediate_steps,
            differential_output_directory=differential_output_directory
        )
        
        image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(symmetry_x, segmented_image)    
        
    return image_data_object

def main():
    image_data_objects = dp.create_image_objects(
        csv_file_path,
        images_directory,
        max_images,
        image_index,
        labels,
        follow_up_number,
        patient_id,
        patient_age,
        patient_gender,
        view_position,
        exclusive_label
    )
    print(f"Created {len(image_data_objects)} image data objects.")
    
    if multiprocessing:
        total_cpus = cpu_count()
        num_workers = max(1, int(total_cpus * core_percentage / 100))
        print(f"Using {num_workers}/{total_cpus} CPU cores...")

        with Pool(processes=num_workers) as pool:
            image_data_objects = pool.map(process_image, image_data_objects)

    else:
        for i, image_data_object in enumerate(image_data_objects):
            image_data_objects[i] = process_image(image_data_object)
         
    if enable_differential_optimization:
        df.differential_optimization(image_data_objects, number_of_trails, max_dataset_size)
        
if __name__ == "__main__":
    main()  # call the main function when the script is run directly