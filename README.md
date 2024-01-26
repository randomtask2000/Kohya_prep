# Kohya LoRA preprocessing

## mov2Images.py
Little image extractor and tagger from selfie video as input. The resulting images are meant to be used by Kohya to create a LoRA for something like SDLX. The images reduced to 512x512.

## resizeImages.py
Resizes images from a directory to an output directory. The images are reduced to 768x768.

# Face Feature Extraction Program Installation & Execution Guide

This guide provides instructions on how to install the necessary dependencies and run the face feature extraction program.

## Dependencies

- Python 3.7 or higher
- OpenCV
- Face Recognition
- NumPy

## Installation

First, ensure that you have Python 3.7 or higher installed on your system. You can download Python from the official website at https://www.python.org/downloads/.

After installing Python, you can install the required Python packages using `pip`. Open a terminal or command prompt and run the following commands:

```sh
pip install opencv-python
pip install face_recognition
pip install numpy
```

## Note
The image or video file path should point to a valid `.png`, `.jpg`, `.jpeg`, or `.mov` file.
The output directory should be a valid directory on your filesystem where the program has write access.
If processing a video, the program will process each frame asynchronously and may take some time to complete, depending on the video length and resolution.

Make sure to copy the Python script into a file named `create.py` and follow the installation guide above to set up your environment. After installation, you'll be able to run the script as instructed.

## Enjoy!
Make this better if you can - cheers!
