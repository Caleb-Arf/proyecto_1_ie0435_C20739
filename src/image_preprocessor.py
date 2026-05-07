from PIL import Image, ImageEnhance
import numpy as np
import os

directories = [
    ("sample_images/cont_positiva", "resized_images/cont_positiva"),
    ("sample_images/cont_negativa", "resized_images/cont_negativa")
]

for input_dir, output_dir in directories:

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert("L")
        mean = np.mean(np.array(img))

        if mean < 100:
            img = ImageEnhance.Brightness(img).enhance(1.3)

        elif mean > 170:
            img = ImageEnhance.Brightness(img).enhance(0.8)

        img = img.resize((128, 128))
        img.save(
            os.path.join(output_dir, filename)
        )

print("Done.")