from pathlib import Path
import os
from PIL import Image
import numpy as np

directories = [
    ("resized_images/cont_positiva", "image_vectors/cont_positiva"),
    ("resized_images/cont_negativa", "image_vectors/cont_negativa")
]

for input_dir, output_dir in directories:

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for img_path in input_dir.glob("*"):

        img = Image.open(img_path).convert("L")
        pixels = np.array(img)

        binary = np.where(pixels < 128, 1, 0)

        output_file = output_dir / (img_path.stem + ".txt")

        with open(output_file, "w") as f:
            for row in binary:
                f.write("".join(map(str, row)) + "\n")

        print(f"Processed: {img_path.name}")