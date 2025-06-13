import numpy as np
import random
import secrets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def run_trial(args):
    trial, atelectasis_group, no_finding_group, min_len, positive_label, resolution, test_size, fixed_threshold = args
    seed = secrets.randbelow(2**32)
    rng = random.Random(seed)
    print(f"Running trial {trial} with seed {seed}...")

    balanced_atelectasis = rng.sample(atelectasis_group, min_len)
    balanced_no_finding = rng.sample(no_finding_group, min_len)
    balanced_result_list = balanced_atelectasis + balanced_no_finding
    rng.shuffle(balanced_result_list)

    result = optimize_thresholds(
        balanced_result_list,
        positive_label=positive_label,
        resolution=resolution,
        test_size=test_size,
        random_seed=seed,
        fixed_threshold = fixed_threshold
    )

    return (
        result['train_accuracy'],
        result['test_accuracy'],
        result['best_config'],
        result['tp'],
        result['tn'],
        result['fp'],
        result['fn']
    )

def optimize_thresholds(result_list, positive_label, resolution=100, test_size=0.2, random_seed=42, fixed_threshold=100):
    symmetry, capacity, labels = [], [], []

    for item in result_list:
        if item['symmetry_percentage'] is not None and item['proportional_lung_capacity'] is not None:
            symmetry.append(item['symmetry_percentage'])
            capacity.append(item['proportional_lung_capacity'])
            
            if positive_label == 'any_disease':
                label_is_positive = item['label'] != 'No Finding'
            else:
                label_is_positive = item['label'] == positive_label

            labels.append(int(label_is_positive))

    X = np.stack([symmetry, capacity], axis=1)
    y = np.array(labels)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    best_acc = 0
    best_config = (None, None, None)

    for w_sp in np.linspace(0, 100, resolution):
        for w_plc in np.linspace(0, 100, resolution):
            scores_train = w_sp * X_train[:, 0] + w_plc * X_train[:, 1]
            scores_test = w_sp * X_test[:, 0] + w_plc * X_test[:, 1]

            if fixed_threshold is not None:
                preds_train = (scores_train > fixed_threshold).astype(int)
                preds_test = (scores_test > fixed_threshold).astype(int)
                acc_train = accuracy_score(y_train, preds_train)
                acc_test = accuracy_score(y_test, preds_test)
                if acc_train > best_acc:
                    best_acc = acc_train
                    best_config = (w_sp, w_plc, fixed_threshold)
            else:
                for thresh in np.linspace(scores_train.min(), scores_train.max(), resolution):
                    preds_train = (scores_train > thresh).astype(int)
                    acc_train = accuracy_score(y_train, preds_train)
                    if acc_train > best_acc:
                        best_acc = acc_train
                        best_config = (w_sp, w_plc, thresh)

    w_sp, w_plc, thresh = best_config
    test_preds = ((w_sp * X_test[:, 0] + w_plc * X_test[:, 1]) > thresh).astype(int)
    test_acc = accuracy_score(y_test, test_preds)

    # Compute TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()

    return {
        'train_accuracy': best_acc,
        'test_accuracy': test_acc,
        'best_config': best_config,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
