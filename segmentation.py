import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def plot_segmentation_image(segmented_lung_image):
    plt.imshow(segmented_lung_image, cmap='gray')
    plt.title("Balck and White Lung Mask")
    plt.axis('off')
    plt.show()

def generate_minimal_lung_image(image_name, image, save_intermediate, intermediate_directory):
    
    # ---------------- Step 0: Normalize brightness to avg 128 ----------------
    mean_val = np.mean(image)
    scale_factor = 128.0 / mean_val
    normalized = np.clip(image.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_0_brightness_normalized.png'), normalized)

    # ---------------- Step 1: CLAHE contrast enhancement ----------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_1_clahe_enhanced.png'), enhanced)

    # ---------------- Step 2: Crop 1% border ----------------
    h, w = enhanced.shape
    margin_h = int(0.01 * h)
    margin_w = int(0.01 * w)
    cropped = enhanced[margin_h:h - margin_h, margin_w:w - margin_w]
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_2_cropped.png'), cropped)

    # ---------------- Step 3: Otsu thresholding ----------------
    _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_3_otsu_threshold.png'), thresh)
    
    # ---------------- Step 4: Flood fill only on outer 15% ----------------
    # Copy before flood fill
    cleared = thresh.copy()
    h_c, w_c = cleared.shape
    mask = np.zeros((h_c + 2, w_c + 2), np.uint8)

    # Define central region
    top = int(0.15 * h_c)
    bottom = int(0.85 * h_c)
    left = int(0.15 * w_c)
    right = int(0.85 * w_c)

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
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_4_cleared_border.png'), cleared)
        
    # ---------------- Step 4.5: Draw center vertical black line ----------------
    center_x = cleared.shape[1] // 2
    cv2.line(cleared, (center_x, 0), (center_x, cleared.shape[0] - 1), color=0, thickness=1)

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_4b_with_center_line.png'), cleared)

    # ---------------- Step 5: Keep the two largest contours ----------------
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    contour_vis = np.zeros_like(cropped)
    cv2.drawContours(contour_vis, contours, -1, (255,), thickness=cv2.FILLED)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_5_lung_contours.png'), contour_vis)

    # ---------------- Step 6: Create lung mask and dilate ----------------
    lung_mask = np.ones_like(cropped, dtype=np.uint8) * 255
    cv2.drawContours(lung_mask, contours, -1, 0, thickness=cv2.FILLED)

    kernel = np.ones((3, 3), np.uint8)
    lung_mask = cv2.dilate(lung_mask, kernel, iterations=1)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_6_lung_mask_dilated.png'), lung_mask)

    # ---------------- Step 7: Draw result and pad back ----------------
    result_cropped = np.ones_like(cropped, dtype=np.uint8) * 255
    cv2.drawContours(result_cropped, contours, -1, (0,), thickness=cv2.FILLED)
    result_cropped = cv2.bitwise_not(result_cropped)

    # Pad cropped result back to original size with black
    result = cv2.copyMakeBorder(result_cropped, margin_h, h - h_c - margin_h, margin_w, w - w_c - margin_w, cv2.BORDER_CONSTANT, value=0)
    
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_7_final_result.png'), result)

    return result
    
def segmentation(image_name, PSO_image_relative_path, output_directory, save_intermediate, intermediate_directory, save_segmentation = True, show_segmentation = False):
    # Load image in grayscale
    PSO_image = cv2.imread(PSO_image_relative_path, cv2.IMREAD_GRAYSCALE)

    segmented_lung_image = generate_minimal_lung_image(image_name, PSO_image, save_intermediate, intermediate_directory)
    
    if save_segmentation:
        cv2.imwrite(f"{output_directory}/{image_name}.png", segmented_lung_image)
        
    if show_segmentation:
        plot_segmentation_image(segmented_lung_image)