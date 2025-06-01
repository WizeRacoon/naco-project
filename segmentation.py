import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def plot_segmentation_image(segmented_lung_image):
    plt.imshow(segmented_lung_image, cmap='gray')
    plt.title("Balck and White Lung Mask")
    plt.axis('off')
    plt.show()

def shade_thresholding(cropped, min_white_pixel_ratio=0.10):
    """
    Find a threshold such that, after applying to the full image, the central 20%–60% vertical region
    (on a 20% cropped version) contains at least one fully black vertical line.
    Falls back to second-to-last gray level if no suitable threshold found.
    """
    h, w = cropped.shape
    total_pixels = h * w

    # Define cropping margins
    margin_h = int(0.2 * h)
    margin_w = int(0.2 * w)
    central = cropped[margin_h:h - margin_h, margin_w:w - margin_w]

    # Define x-range for vertical line check in the central region
    check_x_start = int(0.3 * central.shape[1])
    check_x_end = int(0.7 * central.shape[1])

    # Sorted unique grayscale levels (dark to light)
    unique_vals = sorted(np.unique(cropped))

    for threshold_val in unique_vals:
        _, binary_mask_full = cv2.threshold(cropped, threshold_val, 255, cv2.THRESH_BINARY_INV)

        # Crop binary mask and check for a full black vertical line
        binary_central = binary_mask_full[margin_h:h - margin_h, margin_w:w - margin_w]
        white_ratio_central = np.sum(binary_central == 255) / total_pixels
        
        if not white_ratio_central < min_white_pixel_ratio:     
            for x in range(check_x_start, check_x_end):
                column = binary_central[:, x]
                if np.all(column == 0):  # Fully black line
                    
                    print(f"[INFO] Accepted threshold = {threshold_val} (black vertical line at x={x+(1024*0.2)})")
                    return binary_mask_full

    # Fallback to second-to-last threshold
    if len(unique_vals) >= 2:
        fallback_threshold = unique_vals[-2]
        _, fallback_mask = cv2.threshold(cropped, fallback_threshold, 255, cv2.THRESH_BINARY_INV)
        print(f"[WARNING] No vertical black line found. Fallback to second-to-last threshold = {fallback_threshold}")
        return fallback_mask
    else:
        print("[ERROR] Only one gray level present. Returning full white mask as fallback.")
        return np.ones_like(cropped, dtype=np.uint8) * 255

def flood_fill_border(thresh, fill_limit = 0.15):
    cleared = thresh.copy()
    h_c, w_c = cleared.shape
    mask = np.zeros((h_c + 2, w_c + 2), np.uint8)

    top = int(fill_limit * h_c)
    bottom = int(1 - fill_limit * h_c)
    left = int(fill_limit * w_c)
    right = int(1 - fill_limit * w_c)

    central_region_before = cleared[top:bottom, left:right].copy()

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
    
    cleared[top:bottom, left:right] = central_region_before

    return cleared

def expand_white_pixels(mask, radius=3):
    """
    Expand white pixels by a square neighborhood of size (2*radius + 1).
    """
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return expanded

def generate_minimal_lung_image(image_name, image, save_intermediate, intermediate_directory):
    # ---------------- Step 1: Cut off bottom 90 pixels and crop border 1% ----------------
    h_orig, w_orig = image.shape
    h_cut = h_orig - 90
    image_cut = image[:h_cut, :]

    margin_h = int(0.01 * h_cut)
    margin_w = int(0.01 * w_orig)

    cropped = image_cut[margin_h:h_cut - margin_h, margin_w:w_orig - margin_w]
    h_cropped, w_cropped = cropped.shape

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step1_cropped.png'), cropped)

    # ---------------- Step 2: Shade Thresholding ----------------
    thresh = shade_thresholding(cropped)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step2_shade_threshold.png'), thresh)

    # ---------------- Step 3: Flood Fill Border ----------------
    filled = flood_fill_border(thresh)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step3_flood_fill_border.png'), filled)

    # ---------------- Step 4: Expand White Pixels ----------------
    expanded = expand_white_pixels(filled)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step4_expand_white_pixels.png'), expanded)

    # ---------------- Step 5: Draw Center Vertical Black Line ----------------
    lined = expanded.copy()
    center_x = lined.shape[1] // 2
    cv2.line(lined, (center_x, 0), (center_x, lined.shape[0] - 1), color=0, thickness=1)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step5_center_line.png'), lined)

    # ---------------- Step 6: Keep Two Largest Contours ----------------
    contours, _ = cv2.findContours(lined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    contour_vis = np.zeros_like(cropped)
    cv2.drawContours(contour_vis, contours, -1, (255,), thickness=cv2.FILLED)
    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step6_lung_contours.png'), contour_vis)

    # ---------------- Step 7: Reconstruct Full-Sized Mask ----------------
    result_cropped = np.ones_like(cropped, dtype=np.uint8) * 255
    cv2.drawContours(result_cropped, contours, -1, (0,), thickness=cv2.FILLED)
    result_cropped = cv2.bitwise_not(result_cropped)

    result = cv2.copyMakeBorder(result_cropped,
                                margin_h, h_cut - margin_h - h_cropped,
                                margin_w, w_orig - margin_w - w_cropped,
                                cv2.BORDER_CONSTANT, value=0)

    if save_intermediate:
        cv2.imwrite(os.path.join(intermediate_directory, f'{image_name}_step7_final_result.png'), result)

    return result
    
def segmentation(image_name, PSO_image_relative_path, output_directory, save_intermediate, intermediate_directory, save_segmentation = True, show_segmentation = False):
    # Load image in grayscale
    PSO_image = cv2.imread(PSO_image_relative_path, cv2.IMREAD_GRAYSCALE)

    segmented_lung_image = generate_minimal_lung_image(image_name, PSO_image, save_intermediate, intermediate_directory)
    
    if save_segmentation:
        cv2.imwrite(f"{output_directory}/{image_name}.png", segmented_lung_image)
        
    if show_segmentation:
        plot_segmentation_image(segmented_lung_image)