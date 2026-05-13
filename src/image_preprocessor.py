from PIL import Image, ImageEnhance
import numpy as np
import os

directories = [
    ("sample_images/cont_positiva", "resized_images/cont_positiva"),
    ("sample_images/cont_negativa", "resized_images/cont_negativa")
]

TARGET_MEAN = 128

for input_dir, output_dir in directories:

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert("L")
        img_array = np.array(img, dtype=np.float32)
        current_mean = np.mean(img_array)

        if current_mean > 0:
            brightness_factor = TARGET_MEAN / current_mean
            brightness_factor = max(0.5, min(brightness_factor, 2.0))
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.resize((128, 128))

        img.save(os.path.join(output_dir, filename))

print("Done.")