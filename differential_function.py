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
    
def differential_optimization(image_data_objects, number_of_trails, max_dataset_size = 1000000000, resolution=30, test_size=0.2):
    # Group data by label
    label_groups = defaultdict(list)
    for obj in image_data_objects:
        label = obj.labels[0]
        label_groups[label].append({
            'image_index': obj.image_index,
            'symmetry_percentage': obj.symmetry_percentage,
            'proportional_lung_capacity': obj.proportional_lung_capacity,
            'label': label
        })

    atelectasis_group = label_groups.get('Atelectasis', [])
    no_finding_group = label_groups.get('No Finding', [])
    min_len = min(min(len(atelectasis_group), len(no_finding_group)), max_dataset_size)

    train_accuracies = []
    test_accuracies = []
    best_configs = []

    for trial in range(number_of_trails):
        seed = 42 + trial
        rng = random.Random(seed)

        balanced_atelectasis = rng.sample(atelectasis_group, min_len)
        balanced_no_finding = rng.sample(no_finding_group, min_len)
        balanced_result_list = balanced_atelectasis + balanced_no_finding
        rng.shuffle(balanced_result_list)

        result = oth.optimize_thresholds(
            balanced_result_list,
            positive_label='Atelectasis',
            resolution=resolution,
            test_size=test_size,
            random_seed=seed
        )

        train_accuracies.append(result['train_accuracy'])
        test_accuracies.append(result['test_accuracy'])
        best_configs.append(result['best_config'])

    print(f"Balanced dataset per trial: {min_len} 'Atelectasis' and {min_len} 'No Finding'")
    print(f"train_acc_mean: {np.mean(train_accuracies)}, train_acc_std: {np.std(train_accuracies)}")
    print(f"test_acc_mean: {np.mean(test_accuracies)}, test_acc_std: {np.std(test_accuracies)}")
    for i, (w1, w2, thresh) in enumerate(best_configs):
        print(f"Trial {i+1}: w1={w1}, w2={w2}, threshold={thresh}")