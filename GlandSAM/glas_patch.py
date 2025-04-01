from PIL import Image
import os

def slice_image_fixed(image_path, tile_width, tile_height, output_dir=None):
    img = Image.open(image_path)
    img_width, img_height = img.size

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    os.makedirs(output_dir, exist_ok=True)

    max_x = img_width // tile_width
    max_y = img_height // tile_height

    for y in range(max_y):
        for x in range(max_x):
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height

            tile = img.crop((left, upper, right, lower))

            tile_filename = f"{base_name}_{x}_{y}.png"
            tile_path = os.path.join(output_dir, tile_filename)
            tile.save(tile_path)

    print(f"{max_x * max_y} patches stored at {output_dir}")

proposal_folder_path = 'datasets/Gland Proposal Map/'

for item in os.listdir(proposal_folder_path):
    slice_image_fixed(os.path.join(proposal_folder_path, item), 224, 224, 'datasets/proposal_map')

mask_folder_path = 'datasets/Glas/annotations'

for item in os.listdir(mask_folder_path):
    slice_image_fixed(os.path.join(mask_folder_path, item), 224, 224, 'datasets/masks')

image_folder_path = 'datasets/Glas/image'

for item in os.listdir(image_folder_path):
    slice_image_fixed(os.path.join(image_folder_path, item), 224, 224, 'datasets/images')

pseudo_mask_folder_path = 'datasets/MSSG/pseudo_masks'

for item in os.listdir(pseudo_mask_folder_path):
    slice_image_fixed(os.path.join(pseudo_mask_folder_path, item), 224, 224, 'datasets/pseudo_masks')