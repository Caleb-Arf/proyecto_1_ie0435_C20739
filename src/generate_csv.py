import os
import csv

vector_dir = 'image_vectors'
output_csv = 'vectors.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for filename in os.listdir(vector_dir):
        file_path = os.path.join(vector_dir, filename)
        
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.read().splitlines()
                while lines and lines[-1] == '':
                    lines.pop()
                
                single_line = ''.join(lines)
                
                vector_list = list(single_line)
                
                if len(vector_list) == 16384:
                    writer.writerow(vector_list)
                    print(f"Processed: {filename} - {len(vector_list)} columns")
                else:
                    print(f"Warning: {filename} has {len(vector_list)} chars, expected 16384")