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

exclusive_label                 = False                                     # if 'True': set such that patient only has one label in the list 'labels' | if 'False': set such that a patient can have multiple labels specified

#================================================================================================

# Experiment Parameters =========================================================================
# Here, you can adjust what experiment you want to run and where to put it

### Experiment name------------------
experiment_name                     = "Experiment"

### Differential Optimization--------
differential_optimization           = True                                  # at the end apply differential optimization to the best combination of symmetry_percentage and proportional_lung_capacity

### Multiprocessing------------------
multiprocessing                     = True                                  # apply multiprocessing?
core_percentage                     = 80                                    # percentage of CPU to use when multiprocessing

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
apply_segmentation                  = True

#image_index                        =
#PSO_image_relative_path            =
segmentation_output_directory       = f"{experiment_name}/segmentation/images"
save_intermediate                   = True
segmentation_intermediate_directory = f"{experiment_name}/segmentation/intermediate_steps"


### differential function------------
apply_differential_function         = True

#image_index                        =
#segmented_lung_image_relative_path =
show_symmetry_line                  = False
save_symmetry_line                  = True
symmetry_output_directory           = f"{experiment_name}/symmetry_line/images"
print_findings                      = False
#================================================================================================