import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def optimize_thresholds(result_list, positive_label, resolution=10, test_size=0.2, random_state=42):
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

    # Convert to numpy arrays
    symmetry = np.array(symmetry)
    capacity = np.array(capacity)
    labels = np.array(labels)
    binary_labels = (labels == positive_label).astype(int)

    # Combine into a single array for splitting
    X = np.stack([symmetry, capacity], axis=1)
    y = binary_labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    best_acc = 0
    best_weights = (None, None)
    best_threshold = None

    w1_range = np.linspace(0, 1000, resolution)
    w2_range = np.linspace(0, 1000, resolution)

    for w1 in w1_range:
        for w2 in w2_range:
            scores_train = w1 * X_train[:, 0] + w2 * X_train[:, 1]
            thresholds = np.linspace(scores_train.min(), scores_train.max(), resolution)
            for threshold in thresholds:
                preds_train = (scores_train > threshold).astype(int)
                acc = accuracy_score(y_train, preds_train)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w1, w2)
                    best_threshold = threshold

    # Evaluate on test set using best found params
    w1_best, w2_best = best_weights
    scores_test = w1_best * X_test[:, 0] + w2_best * X_test[:, 1]
    preds_test = (scores_test > best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, preds_test)

    return {
        'train_accuracy': best_acc,
        'test_accuracy': test_accuracy,
        'weight_symmetry': w1_best,
        'weight_capacity': w2_best,
        'threshold': best_threshold
    }