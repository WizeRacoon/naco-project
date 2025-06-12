import os

def count_occurrences_in_directory(directory, csv_file_path):
    # Load labels from Data_Entry_2017.csv
    def load_labels(csv_path):
        labels = {}
        with open(csv_path, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(',')
                image_name = parts[0]
                finding_labels = parts[1]
                view_position = parts[6]
                labels[image_name] = {'labels': finding_labels, 'view_position': view_position}
        return labels

    # Load labels
    labels = load_labels(csv_file_path)

    # Initialize counters
    pa_count = 0
    ap_count = 0
    atelectasis_count = 0
    no_finding_count = 0

    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        if file_name in labels:
            label_info = labels[file_name]
            # Count PA and AP
            if label_info['view_position'] == 'PA':
                pa_count += 1
            elif label_info['view_position'] == 'AP':
                ap_count += 1
            # Count Atelectasis and No Finding
            if 'Atelectasis' in label_info['labels'].split('|'):
                atelectasis_count += 1
            elif label_info['labels'] == 'No Finding':
                no_finding_count += 1

    # Print results
    print(f"Directory: {directory}")
    print(f"  PA: {pa_count}")
    print(f"  AP: {ap_count}")
    print(f"  Atelectasis: {atelectasis_count}")
    print(f"  No Finding: {no_finding_count}")


csv_file_path = 'archive/Data_Entry_2017.csv'  
prefix_path = './only_PA'

# PA-AP train normal
directory_path = prefix_path + '/train/NORMAL' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP train atelectasis
directory_path = prefix_path + '/train/ATELECTASIS' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP val normal
directory_path = prefix_path + '/val/NORMAL' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP val atelectasis
directory_path = prefix_path + '/val/ATELECTASIS' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_1 normal
directory_path = prefix_path + '/test_1/NORMAL' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_1 atelectasis
directory_path = prefix_path + '/test_1/ATELECTASIS' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_2 normal
directory_path = prefix_path + '/test_2/NORMAL' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_2 atelectasis
directory_path = prefix_path + '/test_2/ATELECTASIS' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_3 normal
directory_path = prefix_path + '/test_3/NORMAL' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)

# PA-AP test_3 atelectasis
directory_path = prefix_path + '/test_3/ATELECTASIS' 
print(f"{directory_path=}")
count_occurrences_in_directory(directory_path, csv_file_path)