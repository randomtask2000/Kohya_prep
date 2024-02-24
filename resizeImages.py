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
import face_recognition
import numpy as np
import cv2

def delete_hidden_files_and_directories(directory):
    """
    Recursively deletes hidden files and directories in the specified directory.
    """
    for root, dirs, files in os.walk(directory, topdown=False):
        # Delete hidden files
        for name in files:
            if name.startswith('.'):
                os.remove(os.path.join(root, name))

        # Delete hidden directories
        for name in dirs:
            if name.startswith('.'):
                shutil.rmtree(os.path.join(root, name))


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

def make_square_and_resize(img, size):
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

    # Resize image
    img.thumbnail((size, size))

    return img

# def process_image(input_path, output_path, output_format, size):
#     with Image.open(input_path) as img:
#         img = correct_orientation(img)
#         img = make_square_and_resize(img, size)
#         img.save(output_path, format=output_format)
def get_feature_image(np_image, feature_points):
    """
    Extracts the image segment of a specific facial feature.
    """
    # Create a mask to extract the feature
    mask = np.zeros(np_image.shape[:2], dtype=np.uint8)
    points = np.array(feature_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    feature_image = cv2.bitwise_and(np_image, np_image, mask=mask)
    return feature_image

def analyze_feature_shape_and_size(feature_points):
    """
    Provides a basic analysis of the shape and size of facial features based on their landmarks.
    """
    # Example analysis based on the number of points (simplistic and not very accurate)
    if len(feature_points) > 5:
        shape_description = "complex shape"
    else:
        shape_description = "simple shape"
    size_description = "size cannot be determined without context"  # Placeholder
    return shape_description, size_description

def remove_background_using_grabcut(input_path):
    """
    Use OpenCV's grabCut algorithm to remove the background from an image.
    This function returns an image with a transparent background.
    """
    # Load the image
    img = cv2.imread(input_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define the rectangle area to assume the foreground
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)  # Adjust this rectangle as needed

    # Run grabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # Convert the image to have a transparent background
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    
    return Image.fromarray(dst)

def process_image(lora_name, class_name, input_path, output_path, output_format, size):
    """
    Process the image: correct orientation, make square, resize, make background transparent, and save.
    """
    try:
        with Image.open(input_path) as img:
            img = remove_background_using_grabcut(input_path)
            img = correct_orientation(img)
            img = make_square_and_resize(img, size)
            img.save(output_path, format=output_format)
    except IOError as e:
        print(f"Error processing image: {e}")
    
# def process_image(lora_name, class_name, input_path, output_path, output_format, size):
#     """
#     Process the image: correct orientation, make square, resize, and generate a detailed description.
#     """
#     try:
#         with Image.open(input_path) as img:
#             img = correct_orientation(img)
#             img = make_square_and_resize(img, size)
#             img.save(output_path, format=output_format)
#     except IOError as e:
#         print(f"Error processing image: {e}")

def count_files(directory, allowed_extensions):
    """Counts the number of files in the specified directory with allowed extensions."""
    return len([name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name)) and
                os.path.splitext(name)[1] in allowed_extensions])

def process_images(lora_name, class_name, source_dir, target_dir, size, output_format):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)  # Clear out the output directory
        os.makedirs(target_dir)
    
    # Delete hidden files and directories before processing the images
    delete_hidden_files_and_directories(source_dir)

    allowed_extensions = ['.png', '.jpeg', '.jpg', '.gif']  # List of allowed file extensions
    file_count = count_files(source_dir, allowed_extensions)
    print(f'There are {file_count} files in the directory with allowed extensions.')

    factor = 3
    multiplier = factor * file_count 
    """40_woman"""
    target_dir = os.path.join(target_dir, f"{multiplier}_{class_name}"); os.makedirs(target_dir)
    """dest"""
    target_dir = os.path.join(target_dir, "dest"); os.makedirs(target_dir)
    """40_woman/dest/img"""
    target_dir = os.path.join(target_dir, "img"); os.makedirs(target_dir)
    """40_woman/dest/log"""
    os.makedirs(os.path.join(target_dir, "log"))
    """40_woman/dest/model"""
    os.makedirs(os.path.join(target_dir, "model"))
    """40_woman/dest/40_woman m4rni"""
    target_dir = os.path.join(target_dir, f"{multiplier}_{class_name} {lora_name}")
    os.makedirs(target_dir)

    # copy the images to this dir next and create the text file descriptions and move them there
    print(f'About to write {file_count} images to {target_dir}')
    sequence_number = 1
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if is_valid_image(file_path):
            output_file_name = f"image_{sequence_number}.{output_format.lower()}"
            output_path = os.path.join(target_dir, output_file_name)
            process_image(lora_name, class_name, file_path, output_path, output_format, size)
            sequence_number += 1

# Prompting user for input
default_lora_name = 'M4rni'
lora_name = input(f"Enter lora name (default: {default_lora_name}): ") or default_lora_name.lower()
            
default_class = 'Woman'
class_name = input(f"Enter class name (default: {default_class}): ") or default_class.lower()

default_source_dir = '/dataset/4va9/orig/'
source_dir = input(f"Enter source directory (default: {default_source_dir}): ") or default_source_dir

default_target_dir = '/dataset/4va9/output'
target_dir = input(f"Enter target directory (default: {default_target_dir}): ") or default_target_dir

image_size = int(input("Enter image size (default: 512, or something like 768): ") or 512)

default_output_format = 'PNG'
output_format = input(f"Enter output image format (default: {default_output_format}): ") or default_output_format

process_images(lora_name, class_name, source_dir, default_target_dir, image_size, output_format)



