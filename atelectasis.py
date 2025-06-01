# This is the main file of the project

import data_processor as dp, PSO as pso, segmentation as seg, differential_function as df
import os
from multiprocessing import Pool, cpu_count
import random
from collections import defaultdict
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Data Filters ==================================================================================
# Here, you can adjust which data to consider. If you want to consider all data of a class, just state the list ["any"] (this cannot be done for the "labels" class)

image_index                     = ["any"]                                   # list of indeces of image; unique identifier in the form "'patient_id' + '_' + 'follow_up_number' + '.png'"
labels                          = ["No Finding","Atelectasis"]                           # list of labels (diseases). Possibilities: ["No Finding", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
follow_up_number                = ["any"]                                   # list of number of patients' xray image (e.g, their 25th xray image)
patient_id                      = ["any"]
patient_age                     = ["any"]                                   # list of ages in format 'XXXY', where the 3 X's denote the age, and 'Y' is an abbreviation for year. e.g., '059Y' means the patient is 59 years old
patient_gender                  = ["M"]                                   # list of genders in format ["M","F"]
view_position                   = ["PA"]                                   # list of view positions in format ["PA","AP"]. PA means posteroanterior (image trough the front), AP means anteroposterior (image trough the back)
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

### Differential Optimization--------
enable_differential_optimization    = True                                  # at the end apply differential optimization to the best combination of symmetry_percentage and proportional_lung_capacity

### Multiprocessing------------------
multiprocessing                     = True                                 # apply multiprocessing?
core_percentage                     = 75                                    # percentage of CPU to use when multiprocessing

### Input data location--------------
csv_file_path                       = 'data/input/labels.csv'
images_directory                    = 'data/input/images'

### max amount of images-------------
max_images                          = 10000                               # for if you only want to run a few images

### PSO------------------------------
apply_PSO                           = False

n                                   = 20                                    # number of palettes (particles)
N                                   = 3                                     # number of colors in each palette
max_iterations                      = 20                                    # maximum number of iterations
#image_path                         =
PSO_output_directory                = f"{experiment_name}/PSO/images"
save_iteration_list                 = [1,2,3,4,5,10,20]                     # list of iterations to be saved as an image


### segmentation----------------------------
apply_segmentation                  = True

#image_index                        =
#PSO_image_relative_path            =
segmentation_output_directory       = f"try/segmentation/images"                    # f"{experiment_name}/segmentation/images"
save_intermediate                   = True                                  
segmentation_intermediate_directory = f"try/segmentation/intermediate_steps"        # f"{experiment_name}/segmentation/intermediate_steps"


### differential function------------
apply_differential_function         = True

#image_index                        =
#segmented_lung_image_relative_path =
show_symmetry_line                  = False
save_symmetry_line                  = True
symmetry_output_directory           = f"try/symmetry_line/images"                   # f"{experiment_name}/symmetry_line/images"
print_findings                      = False
#================================================================================================


# Making dirs if they don't exist yet
if apply_PSO:
    os.makedirs(PSO_output_directory, exist_ok=True)
if apply_segmentation:
    os.makedirs(segmentation_output_directory, exist_ok=True)
    if save_intermediate:
        os.makedirs(segmentation_intermediate_directory, exist_ok=True)
if apply_differential_function:
    os.makedirs(symmetry_output_directory, exist_ok=True)

def process_image(image_data_object):
    image_name = image_data_object.image_index.split(".")[0]

    if apply_PSO:
        print(f"[{image_name}] - applying PSO")
        for save_iteration in save_iteration_list:
            if save_iteration < 1 or save_iteration > max_iterations:
                raise ValueError(f"save_iteration_list value {save_iteration} is invalid.")
        pso.PSO(
            n=n,
            N=N,
            image_path=f"{images_directory}/{image_data_object.image_index}",
            max_iterations=max_iterations,
            output_directory=PSO_output_directory,
            save_iteration_list=save_iteration_list
        )

    if apply_segmentation:
        print(f"[{image_name}] - applying segmentation")
        seg.segmentation(
            image_name=image_name,
            PSO_image_relative_path=f"{PSO_output_directory}/{image_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_{max_iterations}.png",
            output_directory=segmentation_output_directory,
            save_intermediate=save_intermediate,
            intermediate_directory=segmentation_intermediate_directory
        )

    if apply_differential_function:
        print(f"[{image_name}] - applying differential function")
        image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(
            image_index=image_data_object.image_index,
            segmented_lung_image_relative_path=f"{segmentation_output_directory}/{image_data_object.image_index}",
            show_symmetry_line=show_symmetry_line,
            save_symmetry_line=save_symmetry_line,
            symmetry_output_directory=symmetry_output_directory,
            print_findings=print_findings
        )
        
    return image_data_object

def differential_optimization(image_data_objects):
    count_atelectasis = 0
    count_no_finding = 0
    for image_data_object in image_data_objects:
        if image_data_object.labels[0] == 'No Finding':
            count_no_finding += 1
        else:
            count_atelectasis += 1
    
    print(f"no findings: {count_no_finding}, atelectasis: {count_atelectasis}")
        
    # Group data by label
    label_groups = defaultdict(list)
    for image_data_object in image_data_objects:
        label = image_data_object.labels[0]
        label_groups[label].append({
            'image_index': image_data_object.image_index,
            'symmetry_percentage': image_data_object.symmetry_percentage,
            'proportional_lung_capacity': image_data_object.proportional_lung_capacity,
            'label': label
        })

    # Get "Atelectasis" and "No Finding" groups
    atelectasis_group = label_groups.get('Atelectasis', [])
    no_finding_group = label_groups.get('No Finding', [])

    # Find the smaller group size
    min_len = min(len(atelectasis_group), len(no_finding_group))

    # Randomly sample both to equal size
    balanced_atelectasis = random.sample(atelectasis_group, min_len)
    balanced_no_finding = random.sample(no_finding_group, min_len)

    # Combine to create balanced result_list
    balanced_result_list = balanced_atelectasis + balanced_no_finding
    random.shuffle(balanced_result_list)  # Optional: shuffle to remove ordering bias

    # Run threshold optimizer
    print(f"Balanced dataset: {len(balanced_atelectasis)} 'Atelectasis' and {len(balanced_no_finding)} 'No Finding'")
    optimized = df.optimize_thresholds(balanced_result_list, positive_label='Atelectasis', resolution=30)
    best_acc = optimized['best_accuracy']
    best_threshold = optimized['threshold']
    weight_symmetry = optimized['weight_symmetry']
    weight_capacity = optimized['weight_capacity']
    print(f"best_accuracy: {best_acc}, weight_symmetry: {weight_symmetry}, weight_capacity: {weight_capacity},threshold: {best_threshold}")
    
    for result in balanced_result_list:
        if result['label'] == 'Atelectasis':
            print(f"{result['image_index']}, {result['label']}, symmetry_percentage: {result['symmetry_percentage']}, proportional_lung_capacity : {result['proportional_lung_capacity']}, {(weight_symmetry * result['symmetry_percentage'] + weight_capacity * result['proportional_lung_capacity']) < best_threshold}")
        

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
        differential_optimization(image_data_objects)
        
if __name__ == "__main__":
    main()  # call the main function when the script is run directly