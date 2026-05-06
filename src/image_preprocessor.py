from PIL import Image, ImageEnhance
import os

input_dir = "sample_images"
output_dir = 'resized_images'


os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        resized_img = img.resize((128, 128))
        resized_img.save(os.path.join(output_dir, filename))

print("Done.")