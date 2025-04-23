import os
from pathlib import Path
from PIL import Image


# Image Resizig
def resize_image_to_1024x1024(input_path, output_path):
    with Image.open(input_path) as img:
        resized_img = img.resize((1024, 1024), Image.ANTIALIAS)
        resized_img.save(output_path)
        print(f"Image saved to {output_path}")


def resize_dataset():
    image_dir = Path("sample/images")
    os.chdir(image_dir)
    print(f"Image dir = {os.getcwd()}")
    for file in os.listdir():
        file_path = Path(file)
        if file_path.is_file():
            resize_image_to_1024x1024(file_path, file_path)

# Making Objects