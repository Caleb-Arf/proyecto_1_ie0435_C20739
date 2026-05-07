import os
import csv

directories = [
    ("training_set/positive_training.csv", "testing_set/positive_test.csv"),
    ("training_set/negative_training.csv", "testing_set/negative_test.csv")
]

os.makedirs("testing_set", exist_ok=True)

for input_csv, test_csv in directories:

    with open(input_csv, 'r') as infile:
        rows = list(csv.reader(infile))

    test_indices = list(range(0, len(rows), 14))[:15]

    test_rows = [rows[i] for i in test_indices]

    remaining_rows = [
        row for i, row in enumerate(rows)
        if i not in test_indices
    ]

    with open(test_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(test_rows)

    with open(input_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(remaining_rows)

    print(f"Generated: {test_csv}")
    print(f"Updated training file: {input_csv}")