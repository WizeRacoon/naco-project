import os
import pandas as pd
from pathlib import Path
from PIL import Image

class ImageData:
    def __init__(self, 
                 image_index, 
                 labels,
                 follow_up_number, 
                 patient_id, 
                 patient_age, 
                 patient_gender, 
                 view_position,
                 original_image_width, 
                 original_image_height, 
                 original_image_pixel_spacing_x, 
                 original_image_pixel_spacing_y, 
                 image_path=None, 
                 image_object=None, 
                 symmetry_percentage=None, 
                 proportional_lung_capacity=None):
        self.image_index = image_index
        self.labels = labels
        self.follow_up_number = follow_up_number 
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

    @property
    def symmetry_percentage(self):
        return self._symmetry_percentage

    @symmetry_percentage.setter
    def symmetry_percentage(self, value):
        self._symmetry_percentage = value

    @property
    def proportional_lung_capacity(self):
        return self._proportional_lung_capacity

    @proportional_lung_capacity.setter
    def proportional_lung_capacity(self, value):
        self._proportional_lung_capacity = value

    def print_details(self):
        print(f"Image Index: {self.image_index}")
        print(f"Labels: {self.labels}")
        print(f"Follow-up #: {self.follow_up_number}")
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
        
def pass_filter(image_index,
                labels,
                follow_up_number,
                patient_id,
                patient_age,
                patient_gender,
                view_position,
                
                filter_image_index,
                filter_labels,
                filter_follow_up_number,
                filter_patient_id,
                filter_patient_age,
                filter_patient_gender,
                filter_view_position,
                
                exclusive_label):   
    if ((filter_image_index[0]        != "any" and image_index        not in filter_image_index     ) or
        (filter_follow_up_number[0]   != "any" and follow_up_number   not in filter_follow_up_number) or
        (filter_patient_id[0]         != "any" and patient_id         not in filter_patient_id      ) or
        (filter_patient_age[0]        != "any" and patient_age        not in filter_patient_age     ) or
        (filter_patient_gender[0]     != "any" and patient_gender     not in filter_patient_gender  ) or
        (filter_view_position[0]      != "any" and view_position      not in filter_view_position   )):
        return False
    
    if exclusive_label:
        if len(labels) > 1:
            return False
        if len(labels) == 1:
            if labels[0] in filter_labels:
                return True
    else:
        for label in labels:
            if label in filter_labels:
                return True
    return False

def create_image_objects(csv_path, 
                         image_folder,
                         max_images,
                         
                         filter_image_index,
                         filter_labels,
                         filter_follow_up_number,
                         filter_patient_id,
                         filter_patient_age,
                         filter_patient_gender,
                         filter_view_position,
                         
                         exclusive_label = False):
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

    for index, row in df.iterrows():
        if len(image_objects) < max_images:
            image_index = row['Image Index']
            labels = row['Finding Labels'].split('|')
            follow_up_number = row['Follow-up #']
            patient_id = row['Patient ID']
            patient_age = row['Patient Age']
            patient_gender = row['Patient Gender']
            view_position = row['View Position']
            original_image_width = row['OriginalImageWidth']
            original_image_height = row['OriginalImageHeight']
            original_image_pixel_spacing_x = row['OriginalImagePixelSpacing_x']
            original_image_pixel_spacing_y = row['OriginalImagePixelSpacing_y']

            # Try to find the corresponding image file
            image_file = None
            for filename in image_filenames:
                if image_index in filename:
                    image_file = filename
                    break

            image_path = None
            image_object = None
            if image_file:
                image_path = os.path.join(image_folder, image_file)
                try:
                    image_object = Image.open(image_path)
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {image_path}")
                except Exception as e:
                    print(f"Warning: Could not open image at {image_path} - {e}")
                    
            symmetry_percentage = None
            proportional_lung_capacity = None
            
            if pass_filter(image_index,
                        labels,
                        follow_up_number,
                        patient_id,
                        patient_age,
                        patient_gender,
                        view_position,
                        filter_image_index,
                        filter_labels,
                        filter_follow_up_number,
                        filter_patient_id,
                        filter_patient_age,
                        filter_patient_gender,
                        filter_view_position,
                        exclusive_label):
                image_data_object = ImageData(
                    image_index=image_index,
                    labels=labels,
                    follow_up_number=follow_up_number,
                    patient_id=patient_id,
                    patient_age=patient_age,
                    patient_gender=patient_gender,
                    view_position=view_position,
                    original_image_width=original_image_width,
                    original_image_height=original_image_height,
                    original_image_pixel_spacing_x=original_image_pixel_spacing_x,
                    original_image_pixel_spacing_y=original_image_pixel_spacing_y,
                    image_path=image_path,
                    image_object=image_object,
                    symmetry_percentage=symmetry_percentage,
                    proportional_lung_capacity=proportional_lung_capacity
                )
                image_objects.append(image_data_object)
    return image_objects