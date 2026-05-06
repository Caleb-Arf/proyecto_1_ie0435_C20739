from pathlib import Path
from PIL import Image
import numpy as np

input_dir = Path("resized_images")
output_dir = Path("image_vectors")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*"):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

        img = Image.open(img_path).convert("L")
        pixels = np.array(img)

        binary = np.where(pixels < 128, 1, 0)

        output_file = output_dir / (img_path.stem + ".txt")

        with open(output_file, "w") as f:
            for row in binary:
                f.write("".join(map(str, row)) + "\n")

        print(f"Processed: {img_path.name}")