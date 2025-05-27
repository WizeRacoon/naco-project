import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def plot_masked_image(lung_mask):
    plt.imshow(lung_mask, cmap='gray')
    plt.title("Balck and White Lung Mask")
    plt.axis('off')
    plt.show()

def generate_minimal_lung_image(image_name, image, save_intermediate, intermediate_directory):
    # Create a mask for flood filling (must be 2 pixels larger than the image)
    
    # Step 1: Use Otsu's method to separate lungs from background
    _,  thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_1_otsu_threshold.png'), thresh)

    # Step 2: Clear border-connected regions
    cleared = thresh.copy()
    h, w = cleared.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for row in range(h):
        for col in [0, w - 1]:
            if cleared[row, col] == 255:
                cv2.floodFill(cleared, mask, (col, row), 0)
    for col in range(w):
        for row in [0, h - 1]:
            if cleared[row, col] == 255:
                cv2.floodFill(cleared, mask, (col, row), 0)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_2_cleared_border.png'), cleared)

    # Step 3: Keep the two largest contours (lungs)
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Create an image to visualize selected contours (optional)
    contour_vis = np.zeros_like(image)
    cv2.drawContours(contour_vis, contours, -1, (255,), thickness=cv2.FILLED)

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_3_lung_contours.png'), contour_vis)

    # Step 4: Create result mask with only lungs (in black)
    lung_mask = np.ones_like(image, dtype=np.uint8) * 255
    cv2.drawContours(lung_mask, contours, -1, 0, thickness=cv2.FILLED)

    # Optional: Slight dilation to recover under-segmented edges
    kernel = np.ones((3, 3), np.uint8)
    lung_mask = cv2.dilate(lung_mask, kernel, iterations=1)

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_4_lung_mask_dilated.png'), lung_mask)

    # Draw only lungs on a clean white canvas
    result = np.ones_like(image, dtype=np.uint8) * 255
    cv2.drawContours(result, contours, -1, (0,), thickness=cv2.FILLED)

    # Invert black and white
    result = cv2.bitwise_not(result)
    cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_5_final_result.png'), result)

    return result
    
def lung_mask(image_name, PSO_image_relative_path, output_directory, save_intermediate, intermediate_directory, save_mask = True, show_mask = False):
    # Load image in grayscale
    PSO_image = cv2.imread(PSO_image_relative_path, cv2.IMREAD_GRAYSCALE)

    lung_mask = generate_minimal_lung_image(image_name, PSO_image, save_intermediate, intermediate_directory)
    
    if save_mask:
        cv2.imwrite(f"{output_directory}/{image_name}.png", lung_mask)
        
    if show_mask:
        plot_masked_image(lung_mask)