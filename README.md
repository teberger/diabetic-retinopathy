# diabetic-retinopathy
Machine Learning project: Kaggle competition for Diabetic Retinopathy detection

## Preprocessing Images
### How-To
The `preprocessing.py` script has a main function that will run through a directory and preprocess every image it encounters. The parameters that need to be set manually are:

- Output image resolution
- Input directory
- Output directory

The `STD_RES` constant defined near the top of the script is the resolution to ouput. For example, `STD_RES = 512` means the output images will be 512x512 pixels. The `__main__` part of the script is at the bottom, and contains two strings that define where to find input images and where to send output images. Modify these to suit your environment.

### The Preprocessing Pipeline
The following is a breakdown of what the preprocessing function does:

1. Converts image to grayscale.
2. Resizes the image.
3. If a notch is present in the image, flips image across both axes.
4. If the image is a "left" image, flips the image across the y-axis.
5. Equalizes the histogram of the image.
