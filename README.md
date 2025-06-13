# naco-project
Title: Particle Swarm Optimization (PSO) for Atelectasis Detection in X-ray Images.\\ 
Authors: Andreea Gabrian, Mika Sipman, and Lisanne Weidmann.\\
Supervisors: Lucia Cavallaro and Inge Wortel

## Source code
The source code can be found in the `src` folder. It contains the following files:
- atelectasis.py: The main file, you run this to run the program.
- data_processor.py: Processes the data from the CSV file and creates the image objects.
- PSO.py: Contains the PSO code as described in the methodology of the report paper.
- differential_function.py: As the name says, this is the differential function method as described in the methodology in the report paper.
- segmentation.py: 'Cuts out' the lungs of a PSO processed image.

### Running the program
Specify the experiment_name, the csv_file_path, and the images_directory. Then run `atelectasis.py`. We recommend to turn on multiprocessing to speed up execution.

#### Parameters
`atelectasis.py` contains a list of parameters you can tweak to define the experiment that you would like to run. Most of the parameters can be left as they are. 

- experiment_name: Defines the name of the experiment.
- csv_file_path: Defines the path where the CSV is placed.
- images_directory: Defines the directory where the images over which you would like to perform PSO are placed.
- apply_pso: Must be set to True if you would like to run PSO and not just the differential function.
- N: Number of colors (set to 3: light, middle, and dark).
- n: Number or particles.
- max_iterations: The number of PSO iterations you would like to perform over an images.
- save_pso_overlay: For every pso image in save_iteration_list; saves the pso overlay as a cutout onto the original image in the folder of that pso image.
- save_iteration_list: Save the resulting iterations from PSO. Indexing starts at 1 (instead of 0).
- apply_differential_function, on_pso_iteration: Apply the differential function on the result of a defined PSO iteration result.   
- use_multiprocessing, core_percentage: Enables multithreading.
- differential_output_directory: Define where to save the differential function output images.
- atelectasis_mode: 'atelectasis_only' or 'any_disease'.
- number_of_trails: The number of times to run the differential optimization (with different random train/test splits).
- max_dataset_size: Max amount of images to use for differential optimization (i.e.; - max_dataset_size = 100 means 100 images of "No Findings" get used and 100 images of the comperative dataset get used).
- resolution: Resolution of the differential optimization
- test_size: Proportion of the dataset to use for testing (i.e.; test_size = 0.2 means 20% of the dataset is used for testing). The complement is used for training.
- fixed_threshold: Fixing the thershold gives a better comparative analysis of the weights and doesn't limit accuracy. If you want a flexible threshold, set it to None.


## Datasets
- Sample dataset (5% of the full dataset): [Link to sample dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- Full dataset (46GB): [Link to full dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

**--- DATASET A ---**

Only 'atelectasis' (and other diseases) - 'no findings' on PA view.

- [Link to only-PA (raw) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa)
- [Link to only-PA (PSO 5 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-pso5)
- [Link to only-PA (PSO 20 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-pso20)

**--- DATASET B ---**

As close as possible 50-50 division on both 'atelectasis' (and other diseases) - 'no findings' and PA-AP view. It's not possible to achieve a perfect 50-50 division, but with count_occurrences.py you can see that it's pretty close.

- [Link to PA-AP_atelectasis-normal (raw) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal)
- [Link to PA-AP_atelectasis-normal (PSO 5 iterations applied) dataset on Kaggle](https://www.kaggle.com/datasets/lisanneweidmann/pa-ap-atelectasis-normal-pso5)

**--- DATASET C ---**

Only images taken from standard position PA, and the images have only the label 'No Findings' or 'Atelectasis' (so no other diseases known to be associated with the patient!). 

- [Link to only-pa-and-atelectasis (raw) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-and-atelectasis)
- [Link to only-pa-and-atelectasis (PSO 5 iterations applied) dataset on Kaggle.](https://www.kaggle.com/datasets/lisanneweidmann/only-pa-atelectasis-pso5)
