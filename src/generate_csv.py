import os
import csv

directories = [
    ("image_vectors/cont_positiva", "training_set/positive_training.csv"),
    ("image_vectors/cont_negativa", "training_set/negative_training.csv")
]

os.makedirs("training_set", exist_ok=True)

for vector_dir, output_csv in directories:

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for filename in os.listdir(vector_dir):
            file_path = os.path.join(vector_dir, filename)

            with open(file_path, 'r') as file:
                lines = file.read().splitlines()
                single_line = ''.join(lines)
                vector_list = list(single_line)
                writer.writerow(vector_list)
                print(f"Processed: {filename}")