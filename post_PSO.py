import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    # Load image in grayscale
    image = cv2.imread('OUTPUT_LUNG_RESULTS/lung/lung_position_clip/velocity_clip/lung_reconstructed_palette_iter_9.png', cv2.IMREAD_GRAYSCALE)

    # Create a mask for flood filling (must be 2 pixels larger than the image)
    h, w = image.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    print("HERE?")

    # Copy the image to avoid modifying original
    flood_filled = image.copy()

    # Define flood fill parameters
    low_diff = 2   # Intensity variation allowed below seed pixel
    high_diff = 2  # Intensity variation allowed above seed pixel
    new_val = 255  # Value to fill with (white)

    print("HERE?")

    # Flood fill from all borders
    for x in range(w):
        cv2.floodFill(flood_filled, mask, (x, 0), new_val, loDiff=low_diff, upDiff=high_diff)
        cv2.floodFill(flood_filled, mask, (x, h - 1), new_val, loDiff=low_diff, upDiff=high_diff)
    for y in range(h):
        cv2.floodFill(flood_filled, mask, (0, y), new_val, loDiff=low_diff, upDiff=high_diff)
        cv2.floodFill(flood_filled, mask, (w - 1, y), new_val, loDiff=low_diff, upDiff=high_diff)

    # Step 1: Adaptive threshold to isolate dark regions (lungs and artifacts)
    _, binary = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)  # Invert so lungs are white

    # Step 2: Remove all white regions connected to the border (not lungs)
    # Use flood fill from borders to remove connected components
    cleared = binary.copy()
    h, w = binary.shape
    mask = np.zeros((h+2, w+2), np.uint8)  # Add padding for floodFill

    # Flood fill all borders (top, bottom, left, right)
    for row in range(h):
        for col in [0, w-1]:  # Left and right edges
            if cleared[row, col] == 255:
                cv2.floodFill(cleared, mask, (col, row), 0)
    for col in range(w):
        for row in [0, h-1]:  # Top and bottom edges
            if cleared[row, col] == 255:
                cv2.floodFill(cleared, mask, (col, row), 0)

    # Step 3: Keep only the two largest remaining white blobs (lungs)
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Step 4: Draw only lungs on a clean white canvas
    result = np.ones_like(image, dtype=np.uint8) * 255
    cv2.drawContours(result, contours, -1, (0,), thickness=cv2.FILLED)

    # Step 5: Show and save
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()

    # Save result
    cv2.imwrite('OUTPUT_LUNG_RESULTS/lung/lung_position_clip/velocity_clip/lung_post_processed_iter_9.png', result)