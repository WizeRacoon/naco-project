from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, find_objects
from sklearn.metrics import accuracy_score

def calc_lung_symmetry_line(binary):
    '''
                # Label connected components (assumes lungs are the 2 largest white blobs)
                labeled, num_features = label(binary)
                
                # Find centroids of components
                centroids = center_of_mass(binary, labeled, range(1, num_features + 1))
                
                # Get x-positions of the two lung centroids
                x_centroids = [c[1] for c in centroids] # I don't really get why there are many more of x_centroids then 2
                
                # Midpoint between two lungs
                symmetry_x = int((x_centroids[0] + x_centroids[1]) / 2)
                
                return symmetry_x
    '''

    # Label connected components
    labeled, num_features = label(binary)

    # Find bounding boxes of each component
    slices = find_objects(labeled)

    # Calculate the center x-position of each bounding box
    x_centers = []
    for sl in slices:
        if sl is None:
            continue
        x_center = (sl[1].start + sl[1].stop) / 2
        x_centers.append(x_center)

    # Pick two largest components and get midpoint of their bounding boxes
    sizes = [(labeled == (i + 1)).sum() for i in range(num_features)]
    largest_indices = np.argsort(sizes)[-2:]
    x_pair = sorted([x_centers[i] for i in largest_indices])
    symmetry_x = int(sum(x_pair) / 2)
    
    return symmetry_x

def plot_symmetry_line(image_index, image, symmetry_x, height, width, show_symmetry_line, save_symmetry_line, symmetry_output_directory):
    
    # Draw red line
    image_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(image_rgb)
    draw.line([(symmetry_x, 0), (symmetry_x, height)], fill=(255, 0, 0), width=5)

    # Show image (if running in Jupyter or local script)
    if show_symmetry_line:
        image_rgb.show()
    if save_symmetry_line:
        image_rgb.save(f"{symmetry_output_directory}/{image_index}")
    
def calc_lung_symmetry_percentage(symmetry_x, binary, height, width):
    # Symmetry calculation around symmetry_x
    count_total = 0
    count_symmetric = 0

    for x in range(0, symmetry_x):
        mirror_x = 2 * symmetry_x - x
        if mirror_x >= width:
            continue
        for y in range(height):
            if binary[y, x] == 1:
                count_total += 1
                if binary[y, mirror_x] == 1:
                    count_symmetric += 1
                    
    symmetry_percentage = (count_symmetric / count_total) if count_total > 0 else 0
    
    return symmetry_percentage

def calc_proportional_lung_capacity(symmetry_x, binary, height, width):
    # The multiple of which the larger lung is larger then the smaller lung
    count_total_left = 0
    count_total_right = 0

    for x in range(0, symmetry_x):
        for y in range(height):
            if binary[y, x] == 1:
                count_total_left += 1
    
    for x in range(symmetry_x, width):
        for y in range(height):
            if binary[y, x] == 1:
                count_total_right += 1
                
    if count_total_left < count_total_right:
        proportional_lung_capacity = count_total_right / count_total_left
    else:
        proportional_lung_capacity = count_total_left / count_total_right
                    
    return proportional_lung_capacity

def calc_lung_symmetry(image_index, segmented_lung_image_relative_path, show_symmetry_line = False, save_symmetry_line = False, symmetry_output_directory = "symmetry_line", print_findings = False):
    # Load and binarize the image
    image = Image.open(segmented_lung_image_relative_path).convert("L")
    binary = np.array(image) > 128
    binary = binary.astype(int)
    height, width = binary.shape
    
    symmetry_x = calc_lung_symmetry_line(binary)
    symmetry_percentage = calc_lung_symmetry_percentage(symmetry_x, binary, height, width)
    proportional_lung_capacity = calc_proportional_lung_capacity(symmetry_x, binary, height, width)
    
    if show_symmetry_line or save_symmetry_line:
        plot_symmetry_line(image_index, image, symmetry_x, height, width, show_symmetry_line, save_symmetry_line, symmetry_output_directory)

    if print_findings:
        print(f"Optimal symmetry line x = {symmetry_x}")
        print(f"Symmetry percentage = {(symmetry_percentage*100):.2f}%")
        print(f"proportional lung capacity = {proportional_lung_capacity}x")
        print("\n")
        
    return symmetry_percentage, proportional_lung_capacity

def optimize_thresholds(result_list, positive_label, resolution=10):
    # Prepare data
    symmetry = []
    capacity = []
    labels = []

    for item in result_list:
        if item['symmetry_percentage'] is None or item['proportional_lung_capacity'] is None:
            continue  # Skip incomplete data
        symmetry.append(item['symmetry_percentage'])
        capacity.append(item['proportional_lung_capacity'])
        labels.append(item['label'])

    symmetry = np.array(symmetry)
    capacity = np.array(capacity)
    labels = np.array(labels)

    # Convert labels to binary (1 = positive, 0 = negative)
    binary_labels = (labels == positive_label).astype(int)

    best_acc = 0
    best_weights = (None, None)
    best_threshold = None

    # Try weight combinations and thresholds
    w1_range = np.linspace(0, 1000, resolution)
    w2_range = np.linspace(0, 1000, resolution)

    for w1 in w1_range:
        for w2 in w2_range:
            scores = w1 * symmetry + w2 * capacity
            thresholds = np.linspace(scores.min(), scores.max(), resolution)
            for threshold in thresholds:
                predictions = (scores > threshold).astype(int)
                acc = accuracy_score(binary_labels, predictions)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w1, w2)
                    best_threshold = threshold

    return {
        'best_accuracy': best_acc,
        'weight_symmetry': best_weights[0],
        'weight_capacity': best_weights[1],
        'threshold': best_threshold
    }