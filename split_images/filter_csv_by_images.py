import os
import csv

# filepath: /home/lisanne/Documents/spring2025/naco-project/filter_csv_by_images.py
def filter_csv_by_images(csv_file, image_folder, output_csv):
    """
    Filters the rows of a CSV file based on whether the images exist in the specified folder.

    Args:
        csv_file (str): Path to the input CSV file.
        image_folder (str): Path to the folder containing images.
        output_csv (str): Path to save the filtered CSV file.
    """
    try:
        # Read the CSV file
        with open(csv_file, mode='r') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            fieldnames = reader.fieldnames

        # Check if images exist in the folder
        filtered_rows = [
            row for row in rows if os.path.exists(os.path.join(image_folder, row['Image Index']))
        ]

        # Write the filtered rows to a new CSV file
        with open(output_csv, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

        print(f"Filtered CSV saved to: {output_csv}")
        print(f"Total entries in original CSV: {len(rows)}")
        print(f"Total entries in filtered CSV: {len(filtered_rows)}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Paths to the input CSV, image folder, and output CSV
    csv_file = "/home/lisanne/Documents/spring2025/naco-project/archive/Data_Entry_2017.csv"
    image_folder = "/home/lisanne/Documents/spring2025/naco-project/images"
    output_csv = "/home/lisanne/Documents/spring2025/naco-project/archive/Filtered_Data_Entry_2017.csv"

    # Call the function
    filter_csv_by_images(csv_file, image_folder, output_csv)