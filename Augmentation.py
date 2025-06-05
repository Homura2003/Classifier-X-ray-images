import os
from PIL import Image
from tqdm import tqdm

def augment_images_with_flips(source_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(source_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                image = Image.open(filepath).convert("RGB")

                subdir = os.path.relpath(root, source_dir)
                output_subdir = os.path.join(output_dir, subdir)
                os.makedirs(output_subdir, exist_ok=True)

                filename_wo_ext, _ = os.path.splitext(file)

                image.save(os.path.join(output_subdir, f"{filename_wo_ext}_orig.jpg"), format='JPEG')

                flipped_v = image.transpose(Image.FLIP_TOP_BOTTOM)
                flipped_v.save(os.path.join(output_subdir, f"{filename_wo_ext}_flipV.jpg"), format='JPEG')

                flipped_h = image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_h.save(os.path.join(output_subdir, f"{filename_wo_ext}_flipH.jpg"), format='JPEG')

                flipped_both = flipped_v.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_both.save(os.path.join(output_subdir, f"{filename_wo_ext}_flipVH.jpg"), format='JPEG')

    print("Augmentatie voltooid.")

source_dir = "D:\\Minor\\Classification\\X-ray\\Oude data\\OG\\Too small"
output_dir = "D:\\Minor\\Classification\\X-ray\\Flip data\\small"

augment_images_with_flips(source_dir, output_dir)

