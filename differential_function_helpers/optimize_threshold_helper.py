import numpy as np
from sklearn.metrics import accuracy_score

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