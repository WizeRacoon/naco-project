import numpy as np
from scipy.ndimage import label, find_objects

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