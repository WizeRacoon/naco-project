#This is the main file of the project

import data_preprocessing as dp, new_PSO as nPSO, post_PSO as pPSO, differential_function as df


'''
# preprocessing data
csv_file_path = 'input_data/labels.csv'
images_directory = 'input_data/images'

image_data_objects = dp.create_image_objects(csv_file_path, images_directory)

print(f"Created {len(image_data_objects)} image data objects.")
print('\n')
image_data_objects[0].print_details() # example
'''

# applying PSO
nPSO.main()

# post processing
pPSO.main()

# applying differential_function