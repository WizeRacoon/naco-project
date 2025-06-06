import numpy as np
import cv2

def shade_thresholding(cropped, min_white_pixel_ratio=0.10):
    """
    Find a threshold such that, after applying to the full image, the central 30%–70% vertical region
    contains at least one fully black vertical line. Among valid thresholds, choose one with highest white pixel ratio.
    Falls back to second-to-last gray level if no suitable threshold found.
    """
    h, w = cropped.shape
    total_pixels = h * w

    # Define margins for central crop
    margin_h = int(0.2 * h)
    margin_w = int(0.2 * w)
    central = cropped[margin_h:h - margin_h, margin_w:w - margin_w]

    check_x_start = int(0.3 * central.shape[1])
    check_x_end = int(0.7 * central.shape[1])
    y_start = int(0.3 * central.shape[0])
    y_end = int(0.7 * central.shape[0])

    unique_vals = sorted(np.unique(cropped), reverse=True)

    valid_thresholds = []

    for threshold_val in unique_vals:
        _, binary_mask_full = cv2.threshold(cropped, threshold_val, 255, cv2.THRESH_BINARY_INV)

        binary_central = binary_mask_full[margin_h:h - margin_h, margin_w:w - margin_w]
        white_ratio_central = np.sum(binary_central == 255) / total_pixels

        if white_ratio_central >= min_white_pixel_ratio:
            for x in range(check_x_start, check_x_end):
                column = binary_central[y_start:y_end, x]
                if np.all(column == 0):  # Valid separation
                    valid_thresholds.append((white_ratio_central, threshold_val, binary_mask_full))
                    break

    # Choose the threshold with the highest white pixel ratio among valid ones
    if valid_thresholds:
        best = max(valid_thresholds, key=lambda x: x[0])  # maximize white coverage
        return best[2]

    # Fallback: use second-to-last threshold value if available
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