# Creating balanced datasets
In order to have perform fair comparisons, we have created scripts for making balanced datasets.

1. First, divide the raw images into 'train', 'val', 'test_1', 'test_2', and 'test_3' subdirectories. These are your baseline images. Depending on how you would like to filter the images, execute either `make_dataset-a.py`, `make_dataset-b.py`, or `make_dataset-c.py`.

(*Optional:*) You can check that the created datasets are balanced by executing `count_occurences.py`. This program counts the amount of images with labels 'No Findings', 'Atelectasis', 'PA', and 'AP' inside directories 'train', 'val', 'test_1', 'test_2', and 'test_3'.

2. Once the raw images are filtered, you might want to perform PSO on them. Our PSO program takes one directory and performs PSO on the images inside this directory. In order to save the images filtered by the previous python scripts into one `all_files` subdirectory, use `collect_all_files.py`.
3. Our PSO program performs PSO for all entries inside a CSV, and otherwise alerts the user that not all entries from the CSV are present in the image dataset folder. Therefore, we have to make sure that the supplied CSV fits the images inside `all_files`. We use `filter_csv_by_images.py` to produce a filtered CSV that contains only the entries we want to perform PSO on.
4. Once the PSO images are produced, you want to order them in the *exact* same way that the baseline dataset is ordered in. You can use `replicate_folder_structure.py` for this purpose. This program takes a 'reference_directory' with the desired folder structure ('train', 'val', 'test_1', 'test_2', and 'test_3') and a 'source_directory' with the processed PSO files, and orders the images from the source directory in the exact same way as the reference directory. The result is stored in the target directory. The ordering is based on the names of the image files, therefore it is important that no files are manually renamed by the user.

(*Optional:*) In order to check how similar resulting images from different PSO generations are. You can take the output folder from the PSO program and run `colors_compare.py` over it. You have to specify inside the python script which PSO generations you want to compare with eachother. 
```python
25: image_5_path = os.path.join(subdir, "lung_reconstructed_palette_iter_5.png")
26: image_20_path = os.path.join(subdir, "lung_reconstructed_palette_iter_20.png")
```