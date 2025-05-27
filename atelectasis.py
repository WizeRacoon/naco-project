# This is the main file of the project

import data_processor as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df, kaggleprep as kp
import os
from math import floor

#================================================================================================
### Experiment name------------------
experiment_name                     = "Experiment-3"


### Input data location--------------
csv_file_path                       = 'data/input/labels.csv'
images_directory                    = 'data/input/images'


### PSO------------------------------
apply_PSO                           = True

n                                   = 20                        # number of palettes (particles)
N                                   = 3                         # number of colors in each palette
max_iterations                      = 5                         # maximum number of iterations
#image_path                         =
PSO_output_directory                = f"{experiment_name}/PSO/images"


### masking----------------------------
apply_masking                       = True

#image_index                        =
#PSO_image_relative_path            =
masking_output_directory            = f"{experiment_name}/lung_mask/images"
save_intermediate                   = False
masking_intermediate_directory      = f"{experiment_name}/lung_mask/intermediate_steps"



### differential function------------
apply_differential_function         = True

#image_index                        =
#masked_lung_image_relative_path    =
show_symmetry_line                  = False
save_symmetry_line                  = True
symmetry_output_directory           = f"{experiment_name}/symmetry_line/images"
print_findings                      = False
#================================================================================================


# Making dirs if they don't exist yet
os.makedirs(PSO_output_directory, exist_ok=True)
os.makedirs(masking_output_directory, exist_ok=True)
if save_intermediate:
    os.makedirs(masking_intermediate_directory, exist_ok=True)
os.makedirs(symmetry_output_directory, exist_ok=True)

    
def main():
    image_data_objects = dp.create_image_objects(csv_file_path, images_directory)
    print(f"Created {len(image_data_objects)} image data objects.")
    
    for image_data_object in image_data_objects:
        if image_data_object.view_position == "PA":
            image_name = image_data_object.image_index.split(".")[0]

            if apply_PSO:
                nPSO.PSO(
                    n=n, 
                    N=N, 
                    image_path= f"{images_directory}/{image_data_object.image_index}", 
                    max_iterations = max_iterations,
                    output_directory = PSO_output_directory)
            
            #pso_images = os.listdir(f"{PSO_output_directory}/{img_name}")
            
            if apply_masking:
                pPSO.lung_mask(
                    image_name = image_name,
                    PSO_image_relative_path = f"{PSO_output_directory}/{image_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_{max_iterations-1}.png", 
                    output_directory = masking_output_directory,
                    save_intermediate = save_intermediate,
                    intermediate_directory = masking_intermediate_directory)
            
            if apply_differential_function: 
                image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(
                    image_index = image_data_object.image_index,
                    masked_lung_image_relative_path = f"{masking_output_directory}/{image_data_object.image_index}",
                    show_symmetry_line = show_symmetry_line,
                    save_symmetry_line = save_symmetry_line, 
                    symmetry_output_directory = symmetry_output_directory, 
                    print_findings = print_findings
                    )

if __name__ == "__main__":
    main()  # call the main function when the script is run directly