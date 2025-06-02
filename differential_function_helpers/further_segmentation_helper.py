import numpy as np
import cv2

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
                    return binary_mask_full

    # Fallback to second-to-last threshold
    if len(unique_vals) >= 2:
        fallback_threshold = unique_vals[-2]
        _, fallback_mask = cv2.threshold(cropped, fallback_threshold, 255, cv2.THRESH_BINARY_INV)
        return fallback_mask
    else:
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