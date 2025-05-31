# This is the main file of the project

import data_processor as dp, PSO as pso, segmentation as seg, differential_function as df
import os
from multiprocessing import Pool, cpu_count

# Data Filters ==================================================================================
# Here, you can adjust which data to consider. If you want to consider all data of a class, just state the list ["any"] (this cannot be done for the "labels" class)

image_index                     = ["any"]                                   # list of indeces of image; unique identifier in the form "'patient_id' + '_' + 'follow_up_number' + '.png'"
labels                          = ["No Finding", 
                                   "Atelectasis", 
                                   "Consolidation", 
                                   "Infiltration", 
                                   "Pneumothorax", 
                                   "Edema", 
                                   "Emphysema", 
                                   "Fibrosis", 
                                   "Effusion", 
                                   "Pneumonia", 
                                   "Pleural_Thickening", 
                                   "Cardiomegaly", 
                                   "Nodule",
                                   "Mass", 
                                   "Hernia"]                                # list of labels (diseases). Possibilities: ["No Finding", "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
follow_up_number                = ["any"]                                   # list of number of patients' xray image (e.g, their 25th xray image)
patient_id                      = ["any"]
patient_age                     = ["any"]                                   # list of ages in format 'XXXY', where the 3 X's denote the age, and 'Y' is an abbreviation for year. e.g., '059Y' means the patient is 59 years old
patient_gender                  = ["any"]                                   # list of genders in format ["M","F"]
view_position                   = ["any"]                                   # list of view positions in format ["PA","AP"]. PA means posteroanterior (image trough the front), AP means anteroposterior (image trough the back)
#OriginalImageWidth             =                                           # trivial as the dataset has converted all images to 1024
#OriginalImageHeight            =                                           # trivial as the dataset has converted all images to 1024
#OriginalImagePixelSpacing_x    =                                           # trivial
#OriginalImagePixelSpacing_y    =                                           # trivial

labels_mutially_exclusive       = False                                     # if 'True': set such that no patient with label A has label B | if 'False': set such that a patient can have multiple labels

#================================================================================================



# Experiment Parameters =========================================================================
# Here, you can adjust what experiment you want to run and where to put it

### Experiment name------------------
experiment_name                     = "Experiment"

### Multiprocessing-=----------------
multiprocessing                     = True                                  # apply multiprocessing?
core_percentage                     = 90                                    # percentage of CPU to use when multiprocessing


### Input data location--------------
csv_file_path                       = 'data/input/labels.csv'
images_directory                    = 'data/input/images'

### max amount of images-------------
max_images                          = 1000000                               # for if you only want to run a few images

### PSO------------------------------
apply_PSO                           = True

n                                   = 20                                    # number of palettes (particles)
N                                   = 3                                     # number of colors in each palette
max_iterations                      = 20                                    # maximum number of iterations
#image_path                         =
PSO_output_directory                = f"{experiment_name}/PSO/images"
save_iteration_list                 = [1,2,3,4,5,10,20]                     # list of iterations to be saved as an image


### segmentation----------------------------
apply_segmentation                  = False

#image_index                        =
#PSO_image_relative_path            =
segmentation_output_directory       = f"{experiment_name}/segmentation/images"
save_intermediate                   = True
segmentation_intermediate_directory = f"{experiment_name}/segmentation/intermediate_steps"


### differential function------------
apply_differential_function         = False

#image_index                        =
#segmented_lung_image_relative_path =
show_symmetry_line                  = False
save_symmetry_line                  = True
symmetry_output_directory           = f"{experiment_name}/symmetry_line/images"
print_findings                      = False
#================================================================================================


# Making dirs if they don't exist yet
if apply_PSO:
    os.makedirs(PSO_output_directory, exist_ok=True)
if apply_segmentation:
    os.makedirs(segmentation_output_directory, exist_ok=True)
    if save_intermediate:
        os.makedirs(segmentation_output_directory, exist_ok=True)
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
        labels_mutially_exclusive
    )
    print(f"Created {len(image_data_objects)} image data objects.")
    
    if multiprocessing:
        total_cpus = cpu_count()
        num_workers = max(1, int(total_cpus * core_percentage/100))
        print(f"Using {num_workers}/{total_cpus} CPU cores...")

        with Pool(processes=num_workers) as pool:
            pool.map(process_image, image_data_objects)
    else:
        for image_data_object in image_data_objects:
            process_image(image_data_object)

if __name__ == "__main__":
    main()  # call the main function when the script is run directly