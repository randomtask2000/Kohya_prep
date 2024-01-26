'''
# Image Processing Utility for Kohya_ss and image preprocessing for LoRA generation.

## Author: Emilio Nicoli

This program is a simple image processing utility that corrects the orientation of 
images, makes them square, and resizes them. It processes a batch of images in a 
specified directory and outputs the processed images to a target directory.

It resizes the images to 768, 768 for AI/SDLX processing for a LoRA with Kohya_ss.

### License
This software is released under the MIT License.

### Dependencies
To run this program, you need to have Python installed on your system along with the following Python libraries:
- Pillow (PIL Fork): This can be installed using pip with the command `pip install Pillow`

### Usage
1. Ensure Python and Pillow are installed.
2. Place this script in a directory.
3. Run the script from the command line or an IDE.
4. Follow the prompts to specify the source and target directories for your images.
'''

import os
import shutil
from PIL import Image, ExifTags

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if it's an image
        return True
    except (IOError, SyntaxError):
        return False

def correct_orientation(img):
    # Correct orientation using EXIF data
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image doesn't have getexif
        pass

    return img

def make_square_and_resize(img, size=(768, 768)):
    # Make the image square by cropping or padding
    width, height = img.size

    # Determine the shorter side
    short_side = min(img.size)

    # Calculate coordinates to create a square image
    left = (width - short_side) / 2
    top = (height - short_side) / 2
    right = (width + short_side) / 2
    bottom = (height + short_side) / 2

    # Crop or pad to make the image square
    img = img.crop((left, top, right, bottom))

    # Resize to 768x768
    img.thumbnail(size)

    return img

def process_image(input_path, output_path):
    with Image.open(input_path) as img:
        img = correct_orientation(img)
        img = make_square_and_resize(img)
        img.save(output_path)

def process_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)  # Clear out the output directory
        os.makedirs(target_dir)

    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if is_valid_image(file_path):
            output_path = os.path.join(target_dir, file_name)
            process_image(file_path, output_path)

# Prompting user for source and target directories
source_dir = input("Enter source directory (default: /ai/LoraImages/): ")
if not source_dir:
    source_dir = '/ai/LoraImages/'

target_dir = input("Enter target directory (default: /ai/LoraImages/output): ")
if not target_dir:
    target_dir = '/ai/LoraImages/output'

process_images(source_dir, target_dir)
