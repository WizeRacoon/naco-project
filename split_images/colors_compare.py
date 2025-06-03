import os
from PIL import Image
import numpy as np

def compare_images_in_subdirectories(root_folder, output_folder):
    """
    Compares `lung_reconstructed_palette_iter_5.png` with `lung_reconstructed_palette_iter_20.png`
    in each subdirectory of the root folder.

    Args:
        root_folder (str): Path to the root folder containing subdirectories.
        output_folder (str): Path to save cropped images and results.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all subdirectories in the root folder
    subdirectories = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    for subdir in subdirectories:
        # Paths to the two images
        image_5_path = os.path.join(subdir, "lung_reconstructed_palette_iter_5.png")
        image_20_path = os.path.join(subdir, "lung_reconstructed_palette_iter_20.png")

        # Check if both images exist
        if not os.path.exists(image_5_path) or not os.path.exists(image_20_path):
            print(f"Skipping subdirectory '{subdir}' as one or both images are missing.")
            continue

        try:
            # Open and process the first image
            with Image.open(image_5_path) as img_5:
                img_5 = img_5.convert("L")  # Convert to grayscale
                width, height = img_5.size
                cropped_img_5 = img_5.crop((0, 0, width, height - 90))  # Crop bottom 90 pixels
                pixel_values_5 = np.array(cropped_img_5)
                unique_colors_5 = np.unique(pixel_values_5)

                # Ensure the first image has exactly 3 grayscale colors
                if len(unique_colors_5) != 3:
                    print(f"Skipping comparison in '{subdir}' as image_5 does not have exactly 3 unique grayscale colors.")
                    continue

                sorted_colors_5 = sorted(unique_colors_5)  # Sort colors from darkest to lightest

            # Open and process the second image
            with Image.open(image_20_path) as img_20:
                img_20 = img_20.convert("L")  # Convert to grayscale
                cropped_img_20 = img_20.crop((0, 0, width, height - 90))  # Crop bottom 90 pixels
                pixel_values_20 = np.array(cropped_img_20)
                unique_colors_20 = np.unique(pixel_values_20)

                # Ensure the second image has exactly 3 grayscale colors
                if len(unique_colors_20) != 3:
                    print(f"Skipping comparison in '{subdir}' as image_20 does not have exactly 3 unique grayscale colors.")
                    continue

                sorted_colors_20 = sorted(unique_colors_20)  # Sort colors from darkest to lightest

            # Compare pixel categories
            comparison_result = np.zeros_like(pixel_values_5, dtype=bool)
            for i in range(pixel_values_5.shape[0]):
                for j in range(pixel_values_5.shape[1]):
                    pixel_5 = pixel_values_5[i, j]
                    pixel_20 = pixel_values_20[i, j]

                    # Map pixel_5 to its category
                    if pixel_5 == sorted_colors_5[0]:
                        category_5 = "darkest"
                    elif pixel_5 == sorted_colors_5[1]:
                        category_5 = "intermediate"
                    else:
                        category_5 = "lightest"

                    # Map pixel_20 to its category
                    if pixel_20 == sorted_colors_20[0]:
                        category_20 = "darkest"
                    elif pixel_20 == sorted_colors_20[1]:
                        category_20 = "intermediate"
                    else:
                        category_20 = "lightest"

                    # Check if categories match
                    comparison_result[i, j] = (category_5 == category_20)

            # Calculate match percentage
            match_percentage = np.mean(comparison_result) * 100
            print(f"Comparison for subdirectory '{subdir}': {match_percentage:.2f}% pixels match.")

            # Save cropped images for debugging
            cropped_img_5.save(os.path.join(output_folder, f"{os.path.basename(subdir)}_cropped_iter_5.png"))
            cropped_img_20.save(os.path.join(output_folder, f"{os.path.basename(subdir)}_cropped_iter_20.png"))

        except Exception as e:
            print(f"Error processing subdirectory '{subdir}': {e}")


# Example usage
if __name__ == "__main__":
    root_folder = "results_toy-samples"  
    output_folder = "results_toy-samples/cropped_images"  

    compare_images_in_subdirectories(root_folder, output_folder)