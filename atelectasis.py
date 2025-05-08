# This is the main file of the project

import data_processor as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df, kaggleprep as kp
import os
from math import floor

def main():
    # preparting data for kaggle
    # kp.main() 

    '''
    # Preprocessing data
    csv_file_path = 'data/input/labels.csv'
    images_directory = 'data/input/images'
    
    image_data_objects = dp.create_image_objects(csv_file_path, images_directory)
    print(f"Created {len(image_data_objects)} image data objects.")
    print('\n')
    image_data_objects[0].print_details() # example

    for image_data_object in image_data_objects:
        img_name = image_data_object.image_index.split(".")[0]

        nPSO.PSO(n=20, N=3, image_path= f"{images_directory}/{image_data_object.image_index}", max_iterations=10)
        pPSO.lung_mask(image_index = image_data_object.image_index, 
                       PSO_image_relative_path = f"data/PSO/images/{img_name}/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_9.png", 
                       output_directory = "data/masked_lungs/images")
        image_data_object.symmetry_percentage, image_data_object.proportional_lung_capacity = df.calc_lung_symmetry(f"data/masked_lungs/images/{image_data_object.image_index}", show_symmetry_line=False)
        
        image_data_object.print_details()
    '''

    df.calc_lung_symmetry(f"OUTPUT_LUNG_RESULTS/lung/lung_position_clip/velocity_clip/lung_post_processed_iter_9.png", show_symmetry_line=True)
    df.calc_lung_symmetry(f"OUTPUT_LUNG_RESULTS/lung2/lung_position_clip/velocity_clip/lung_post_processed_iter_9.png", show_symmetry_line=True)

if __name__ == "__main__":
    main()  # call the main function when the script is run directly