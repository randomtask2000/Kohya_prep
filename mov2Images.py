'''
# This program extracts images of faces from a video and tags them for LoRA generation with Kohya_ss

Little image extractor and tagger from selfie video as input. 
The resulting images are meant to be used by Kohya to create a LoRA for something like SDLX.

## Author

Emilio Nicoli

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Installation

Before running the program, ensure you have Python installed on your system. If not, you can download it from [python.org](https://www.python.org/downloads/).

### Dependencies

This program requires the following dependencies:
- Dependency1
- Dependency2
- Dependency3

You can install these dependencies by running the following command:

`pip install Dependency1 Dependency2 Dependency3`
'''
import cv2
import os
import random
import face_recognition
import numpy as np

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

def resize_and_pad(image, target_size=(512, 512)):
    original_size = image.shape[:2]
    ratio = float(target_size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    pad_h = (target_size[0] - new_size[0]) // 2
    pad_w = (target_size[1] - new_size[1]) // 2
    padding_color = random_color()
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image

def get_head_crop(frame, face_location):
    top, right, bottom, left = face_location
    
    head_top = int(max(0, top - (bottom - top) * 0.5))
    head_bottom = int(min(frame.shape[0], bottom + (bottom - top) * 0.25))
    head_left = int(max(0, left - (right - left) * 0.1))
    head_right = int(min(frame.shape[1], right + (right - left) * 0.1))

    head_region = frame[head_top:head_bottom, head_left:head_right]
    head_crop = resize_and_pad(head_region)
    return head_crop

def get_tags_from_landmarks(face_landmarks):
    feature_tags = []
    for feature in face_landmarks.keys():
        feature_tags.append(feature)
    return feature_tags

def get_random_face_crop_and_tags(frame, face_locations):
    if not face_locations:
        return None, [], None, []

    face_location = random.choice(face_locations)
    top, right, bottom, left = face_location

    face_crop = frame[top:bottom, left:right]
    face_landmarks_list = face_recognition.face_landmarks(face_crop)
    
    # If facial features are detected, use them to generate tags
    if face_landmarks_list:
        feature_tags = get_tags_from_landmarks(face_landmarks_list[0])
    else:
        # Fallback mechanism to generate generic tags
        feature_tags = ['face', 'forehead', 'eye_region', 'nose_region', 'mouth_region', 'chin_region']

    entire_head_crop = get_head_crop(frame, face_location)
    head_tags = ["head", "face"] + feature_tags

    return face_crop, feature_tags, entire_head_crop, head_tags

# Input parameters for file path and output directory
file_path = input("Enter the path to your file (image or video): ")
output_dir = input("Enter the output directory for the cropped images: ")

# Check and create default output directory if necessary
if not output_dir:
    output_dir = "default_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if os.path.exists(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Image file '{file_path}' not found.")
            exit()
        
        rgb_image = image[:, :, ::-1]  # Convert to RGB
        face_locations = face_recognition.face_locations(rgb_image)
        feature_crop, feature_tags, entire_head_crop, head_tags = get_random_face_crop_and_tags(rgb_image, face_locations)

        if feature_crop is not None and entire_head_crop is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            feature_file_path = os.path.join(output_dir, 'feature_crop.png')
            head_file_path = os.path.join(output_dir, 'head_crop.png')
            
            cv2.imwrite(feature_file_path, feature_crop[:, :, ::-1])  # Convert back to BGR
            cv2.imwrite(head_file_path, entire_head_crop[:, :, ::-1])
            
            print(f"Tags for feature crop: {feature_tags}")
            print(f"Tags for head crop: {head_tags}")
            print(f"Saved cropped images to {output_dir}")
        else:
            print("No face detected in the image.")
    elif file_path.lower().endswith('.mov'):
        cap = cv2.VideoCapture(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]  # Convert to RGB
            face_locations = face_recognition.face_locations(rgb_frame)
            feature_crop, feature_tags, entire_head_crop, head_tags = get_random_face_crop_and_tags(rgb_frame, face_locations)

            if feature_crop is not None and entire_head_crop is not None:
                feature_file_path = os.path.join(output_dir, f'feature_crop_{frame_count}.png')
                head_file_path = os.path.join(output_dir, f'head_crop_{frame_count}.png')
                
                cv2.imwrite(feature_file_path, feature_crop[:, :, ::-1])  # Convert back to BGR
                cv2.imwrite(head_file_path, entire_head_crop[:, :, ::-1])

                print(f"Saved {feature_file_path} with tags: {feature_tags}")
                print(f"Saved {head_file_path} with tags: {head_tags}")

                frame_count += 1
        cap.release()
else:
    print(f"Error: File '{file_path}' not found.")
