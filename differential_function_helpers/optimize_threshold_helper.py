import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def optimize_thresholds(result_list, positive_label, resolution=10, test_size=0.2, random_seed=42):
    # Prepare data
    symmetry, capacity, labels = [], [], []

    for item in result_list:
        if item['symmetry_percentage'] is not None and item['proportional_lung_capacity'] is not None:
            symmetry.append(item['symmetry_percentage'])
            capacity.append(item['proportional_lung_capacity'])
            labels.append(item['label'])

    X = np.stack([symmetry, capacity], axis=1)
    y = (np.array(labels) == positive_label).astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    best_acc = 0
    best_config = (None, None, None)

    for w1 in np.linspace(0, 1000, resolution):
        for w2 in np.linspace(0, 1000, resolution):
            scores = w1 * X_train[:, 0] + w2 * X_train[:, 1]
            for thresh in np.linspace(scores.min(), scores.max(), resolution):
                preds = (scores > thresh).astype(int)
                acc = accuracy_score(y_train, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_config = (w1, w2, thresh)

    # Evaluate on test set
    w1, w2, thresh = best_config
    test_preds = ((w1 * X_test[:, 0] + w2 * X_test[:, 1]) > thresh).astype(int)
    test_acc = accuracy_score(y_test, test_preds)

    return {
        'train_accuracy': best_acc,
        'test_accuracy': test_acc,
        'best_config': best_config
    }