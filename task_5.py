###############################################
# Author: Amr Ali 
# Date: 21/10
# Version: 1.0
###############################################
# Description: 
# This script creates a 256x256 grayscale gradient image using NumPy and OpenCV. 
# The gradient transitions from white at the top (value 255) to black at the bottom (value 0).
# The image is displayed using OpenCV and closes when a key is pressed.
#
# Usage: 
# - Run this script using a Python environment with OpenCV and NumPy installed.
# - Ensure that OpenCV's GUI functions (e.g., cv2.imshow) are supported in your environment.
#
# Dependencies: 
# - Python 3.x
# - OpenCV (cv2)
# - NumPy (for array manipulation)
#
# Input: 
# - No external input; the script generates the gradient image.
#
# Output: 
# - Displays a 256x256 grayscale gradient image that transitions from white to black.
###############################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create an empty array for the image
img = np.zeros((256, 256), dtype=np.uint8)

# Use a single loop to fill the gradient
for i in range(256):
    img[i, :] = 255 - i  # Set the same value across the row

# Display the image using OpenCV
cv2.imshow('', img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
