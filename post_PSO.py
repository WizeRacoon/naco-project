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
    os.makedirs(intermediate_directory, exist_ok=True)

    # ---------------- Step 0: CLAHE contrast enhancement ----------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_0_clahe_enhanced.png'), enhanced)

    # ---------------- Step 1: Crop 1% border ----------------
    h, w = enhanced.shape
    margin_h = int(0.01 * h)
    margin_w = int(0.01 * w)
    cropped = enhanced[margin_h:h - margin_h, margin_w:w - margin_w]
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_1_cropped.png'), cropped)

    # ---------------- Step 2: Otsu thresholding ----------------
    _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_2_otsu_threshold.png'), thresh)

    # ---------------- Step 3: Flood fill only on outer 20% ----------------
    # Copy before flood fill
    cleared = thresh.copy()
    h_c, w_c = cleared.shape
    mask = np.zeros((h_c + 2, w_c + 2), np.uint8)

    # Define central region
    top = int(0.2 * h_c)
    bottom = int(0.8 * h_c)
    left = int(0.2 * w_c)
    right = int(0.8 * w_c)

    # Extract central region before flood fill
    central_region_before = cleared[top:bottom, left:right].copy()

    # --- Perform unrestricted border flood fill ---
    for col in range(w_c):
        if cleared[0, col] == 255:
            cv2.floodFill(cleared, mask, (col, 0), 0)
        if cleared[h_c - 1, col] == 255:
            cv2.floodFill(cleared, mask, (col, h_c - 1), 0)

    for row in range(h_c):
        if cleared[row, 0] == 255:
            cv2.floodFill(cleared, mask, (0, row), 0)
        if cleared[row, w_c - 1] == 255:
            cv2.floodFill(cleared, mask, (w_c - 1, row), 0)

    # Restore central region
    cleared[top:bottom, left:right] = central_region_before

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_3_cleared_border.png'), cleared)

    # ---------------- Step 4: Keep the two largest contours ----------------
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    contour_vis = np.zeros_like(cropped)
    cv2.drawContours(contour_vis, contours, -1, (255,), thickness=cv2.FILLED)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_4_lung_contours.png'), contour_vis)

    # ---------------- Step 5: Create lung mask and dilate ----------------
    lung_mask = np.ones_like(cropped, dtype=np.uint8) * 255
    cv2.drawContours(lung_mask, contours, -1, 0, thickness=cv2.FILLED)

    kernel = np.ones((3, 3), np.uint8)
    lung_mask = cv2.dilate(lung_mask, kernel, iterations=1)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_5_lung_mask_dilated.png'), lung_mask)

    # ---------------- Step 6: Draw result and pad back ----------------
    result_cropped = np.ones_like(cropped, dtype=np.uint8) * 255
    cv2.drawContours(result_cropped, contours, -1, (0,), thickness=cv2.FILLED)
    result_cropped = cv2.bitwise_not(result_cropped)

    # Pad cropped result back to original size with black
    result = cv2.copyMakeBorder(result_cropped, margin_h, h - h_c - margin_h, margin_w, w - w_c - margin_w, cv2.BORDER_CONSTANT, value=0)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_6_final_result.png'), result)

    return result
    
def lung_mask(image_name, PSO_image_relative_path, output_directory, save_intermediate, intermediate_directory, save_mask = True, show_mask = False):
    # Load image in grayscale
    PSO_image = cv2.imread(PSO_image_relative_path, cv2.IMREAD_GRAYSCALE)

    lung_mask = generate_minimal_lung_image(image_name, PSO_image, save_intermediate, intermediate_directory)
    
    if save_mask:
        cv2.imwrite(f"{output_directory}/{image_name}.png", lung_mask)
        
    if show_mask:
        plot_masked_image(lung_mask)