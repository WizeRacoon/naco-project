import differential_function_helpers.further_segmentation_helper as fsh, differential_function_helpers.lung_symmetry_helper as lsh, differential_function_helpers.optimize_threshold_helper as oth
import os
import cv2
import numpy as np
import random
from collections import defaultdict

def full_segmentation(image_name, image, save_symmetry_line, save_intermediate_steps, differential_directory):
    # ---------------- Step 1: Cut off bottom 90 pixels ----------------
    h_orig, w_orig = image.shape
    h_cut = h_orig - 90
    image_cut = image[:h_cut, :]
    
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step1_cut.png'), image_cut)

    # ---------------- Step 2: Shade Thresholding ----------------
    thresh = fsh.shade_thresholding(image_cut)
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step2_shade_threshold.png'), thresh)


    # ---------------- Step 3: Flood Fill Border with 1% Blacked Out ----------------
    margin_h = int(0.01 * h_cut)
    margin_w = int(0.01 * w_orig)

    thresh_masked = thresh.copy()
    # Black out 1% margins
    thresh_masked[:margin_h, :] = 255
    thresh_masked[-margin_h:, :] = 255
    thresh_masked[:, :margin_w] = 255
    thresh_masked[:, -margin_w:] = 255

    filled = fsh.flood_fill_border(thresh_masked)
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step3_flood_fill_border.png'), filled)

    # ---------------- Step 4: Expand White Pixels ----------------
    expanded = fsh.expand_white_pixels(filled, 3)
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step4_expand_white_pixels.png'), expanded)

    # ---------------- Step 5: Draw Center Vertical Black Line ----------------
    lined = expanded.copy()
    center_x = lined.shape[1] // 2
    cv2.line(lined, (center_x, 0), (center_x, lined.shape[0] - 1), color=0, thickness=1)
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step5_center_line.png'), lined)

    # ---------------- Step 6: Keep Two Largest Contours ----------------
    contours, _ = cv2.findContours(lined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    contour_vis = np.zeros_like(image_cut)
    cv2.drawContours(contour_vis, contours, -1, (255,), thickness=cv2.FILLED)
    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step6_lung_contours.png'), contour_vis)

    # ---------------- Step 7: Segmentation Result ----------------
    result = np.ones_like(image_cut, dtype=np.uint8) * 255
    cv2.drawContours(result, contours, -1, (0,), thickness=cv2.FILLED)
    result = cv2.bitwise_not(result)

    if save_intermediate_steps:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step7_segmentation_result.png'), result)
        
    # ---------------- Step 8: Lung Symmetry ----------------
    binary = np.array(result) > 128
    binary = binary.astype(int)
    height, width = binary.shape

    symmetry_x = lsh.calc_lung_symmetry_line(binary)

    # Create a color version of the result image for drawing in color
    symmetry_line = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Draw vertical red line of 5 pixels wide centered at symmetry_x
    for x in range(symmetry_x - 2, symmetry_x + 3):
        if 0 <= x < width:
            cv2.line(symmetry_line, (x, 0), (x, height - 1), (0, 0, 255), 1)

    if save_intermediate_steps or save_symmetry_line:
        cv2.imwrite(os.path.join(differential_directory, f'{image_name}_step8_symmetry_line.png'), symmetry_line)

    return symmetry_x, result
    
def further_segmentation(image_name, pso_image_relative_path, save_symmetry_line, save_intermediate_steps, differential_output_directory):
    # Load image in grayscale
    pso_image = cv2.imread(pso_image_relative_path, cv2.IMREAD_GRAYSCALE)

    return full_segmentation(image_name, pso_image, save_symmetry_line, save_intermediate_steps, differential_output_directory)

def calc_lung_symmetry(symmetry_x, segmented_image):
    # Load and binarize the image
    binary = segmented_image > 128
    binary = binary.astype(int)
    height, width = binary.shape
    
    symmetry_percentage = lsh.calc_lung_symmetry_percentage(symmetry_x, binary, height, width)
    proportional_lung_capacity = lsh.calc_proportional_lung_capacity(symmetry_x, binary, height, width)
   
    return symmetry_percentage, proportional_lung_capacity
    
def differential_optimization(image_data_objects):
    count_atelectasis = 0
    count_no_finding = 0
    for image_data_object in image_data_objects:
        if image_data_object.labels[0] == 'No Finding':
            count_no_finding += 1
        if len(image_data_object.labels) == 1 and image_data_object.labels[0] == 'Atelectasis':
            count_atelectasis += 1
    
    print(f"no findings: {count_no_finding}, atelectasis: {count_atelectasis}")
        
    # Group data by label
    label_groups = defaultdict(list)
    for image_data_object in image_data_objects:
        label = image_data_object.labels[0]
        label_groups[label].append({
            'image_index': image_data_object.image_index,
            'symmetry_percentage': image_data_object.symmetry_percentage,
            'proportional_lung_capacity': image_data_object.proportional_lung_capacity,
            'label': label
        })

    # Get "Atelectasis" and "No Finding" groups
    atelectasis_group = label_groups.get('Atelectasis', [])
    no_finding_group = label_groups.get('No Finding', [])

    # Find the smaller group size
    min_len = min(len(atelectasis_group), len(no_finding_group))

    # Randomly sample both to equal size
    balanced_atelectasis = random.sample(atelectasis_group, min_len)
    balanced_no_finding = random.sample(no_finding_group, min_len)

    # Combine to create balanced result_list
    balanced_result_list = balanced_atelectasis + balanced_no_finding
    random.shuffle(balanced_result_list)  # Optional: shuffle to remove ordering bias

    # Run threshold optimizer
    print(f"Balanced dataset: {len(balanced_atelectasis)} 'Atelectasis' and {len(balanced_no_finding)} 'No Finding'")
    optimized = oth.optimize_thresholds(balanced_result_list, positive_label='Atelectasis', resolution=30)
    train_acc = optimized['train_accuracy']
    test_acc = optimized['test_accuracy']
    best_threshold = optimized['threshold']
    weight_symmetry = optimized['weight_symmetry']
    weight_capacity = optimized['weight_capacity']
    print(f"train_accuracy: {train_acc}, test_accuracy: {test_acc}, weight_symmetry: {weight_symmetry}, weight_capacity: {weight_capacity},threshold: {best_threshold}")