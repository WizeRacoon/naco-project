import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_masked_image(lung_mask):
    plt.imshow(lung_mask, cmap='gray')
    plt.title("Black and White Lung Mask")
    plt.axis('off')
    plt.show()

def generate_minimal_lung_image(image):
    # Optional: Improve contrast using histogram equalization
    image_eq = cv2.equalizeHist(image)

    # Step 1: Use Otsu's method to threshold
    _, thresh_otsu = cv2.threshold(image_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Fallback if too aggressive (e.g. all black or white)
    black_ratio = np.mean(thresh_otsu == 0)
    if black_ratio > 0.95 or black_ratio < 0.05:
        # Adaptive threshold as fallback
        thresh = cv2.adaptiveThreshold(image_eq, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       11, 2)
    else:
        thresh = thresh_otsu

    # Step 3: Remove border-touching regions
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

    # Step 4: Morphological cleanup to remove small specks and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(cleared, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 5: Keep the two largest contours (lungs)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Step 6: Create result mask
    result = np.ones_like(image, dtype=np.uint8) * 255
    cv2.drawContours(result, contours, -1, (0,), thickness=cv2.FILLED)

    # Invert so lungs appear black
    result = cv2.bitwise_not(result)

    return result

def lung_mask(image_index, PSO_image_relative_path, output_directory, save_mask=True, show_mask=False):
    PSO_image = cv2.imread(PSO_image_relative_path, cv2.IMREAD_GRAYSCALE)
    lung_mask = generate_minimal_lung_image(PSO_image)

    if save_mask:
        cv2.imwrite(f"{output_directory}/{image_index}", lung_mask)

    if show_mask:
        plot_masked_image(lung_mask)