import os
import pandas as pd
from pathlib import Path
from PIL import Image

class ImageData:
    def __init__(self, image_index, labels, patient_id, patient_age, patient_gender, view_position, original_image_width, original_image_height, original_image_pixel_spacing_x, original_image_pixel_spacing_y, image_path=None, image_object=None, symmetry_percentage=None, proportional_lung_capacity=None):
        self.image_index = image_index
        self.labels = labels
        self.patient_id = patient_id
        self.patient_age = patient_age
        self.patient_gender = patient_gender
        self.view_position = view_position
        self.original_image_width = original_image_width
        self.original_image_height = original_image_height
        self.original_image_pixel_spacing_x = original_image_pixel_spacing_x
        self.original_image_pixel_spacing_y = original_image_pixel_spacing_y
        self.image_path = image_path
        self.image_object = image_object
        self.symmetry_percentage = symmetry_percentage
        self.proportional_lung_capacity = proportional_lung_capacity

    def __repr__(self):
        return f"ImageData(image_index='{self.image_index}', patient_id='{self.patient_id}')"

    def print_details(self):
        print(f"Image Index: {self.image_index}")
        print(f"Labels: {self.labels}")
        print(f"Patient ID: {self.patient_id}")
        print(f"Patient Age: {self.patient_age}")
        print(f"Patient Gender: {self.patient_gender}")
        print(f"View Position: {self.view_position}")
        print(f"Original Image Width: {self.original_image_width}")
        print(f"Original Image Height: {self.original_image_height}")
        print(f"Original Image Pixel Spacing x: {self.original_image_pixel_spacing_x}")
        print(f"Original Image Pixel Spacing y: {self.original_image_pixel_spacing_y}")
        if self.image_path:
            print(f"Image Path: {self.image_path}")
        if self.image_object:
            print(f"Image Size: {self.image_object.size}")
        if self.symmetry_percentage:
            print(f"Symmetry percentage (*100): {(self.symmetry_percentage*100):.2f}%")
        if self.proportional_lung_capacity:
            print(f"The larger lung is {self.proportional_lung_capacity}x larger then the smaller lung")
        print("-" * 20)

def create_image_objects(csv_path, image_folder):
    """
    Reads the CSV file and creates ImageData objects, attempting to associate
    them with image files in the specified folder.

    Args:
        csv_path (str): The path to the CSV file.
        image_folder (str): The path to the folder containing the images.

    Returns:
        list: A list of ImageData objects.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []

    image_objects = []
    image_filenames = set(os.listdir(image_folder))

    for _, row in df.iterrows():
        image_index = row['Image Index']
        labels = row['Finding Labels']
        patient_id = row['Patient ID']
        patient_age = row['Patient Age']
        patient_gender = row['Patient Gender']
        view_position = row['View Position']
        original_image_width = row['OriginalImageWidth']
        original_image_height = row['OriginalImageHeight']
        original_image_pixel_spacing_x = row['OriginalImagePixelSpacing_x']
        original_image_pixel_spacing_y = row['OriginalImagePixelSpacing_y']

        # Try to find the corresponding image file
        image_file = next((fname for fname in image_filenames if image_index in fname), None)

        if image_file:
            image_path = os.path.join(image_folder, image_file)
            try:
                image_object = Image.open(image_path)

                image_data_object = ImageData(
                    image_index=image_index,
                    labels=labels,
                    patient_id=patient_id,
                    patient_age=patient_age,
                    patient_gender=patient_gender,
                    view_position=view_position,
                    original_image_width=original_image_width,
                    original_image_height=original_image_height,
                    original_image_pixel_spacing_x=original_image_pixel_spacing_x,
                    original_image_pixel_spacing_y=original_image_pixel_spacing_y,
                    image_path=image_path,
                    image_object=image_object
                )
                image_objects.append(image_data_object)

            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not open image at {image_path} - {e}")
                continue  # Skip creation if image can't be opened

    return image_objects