
###############################################
# Author: Amr Ali 
# Date: 20/10
# Version: 1.0
###############################################
# Description: 
# This script creates a blank 256x256 pixel image with a white background 
# and draws two black squares: one in the top-right corner and another in 
# the bottom-left corner. The image is then displayed using OpenCV.
#
# Usage: 
# - Simply run the script in a Python environment with OpenCV and NumPy installed.
#
# Dependencies: 
# - Python 3.x
# - OpenCV (cv2)
# - NumPy (for array manipulations)
#
# Input: 
# - No external image input is required.
#
# Output: 
# - Displays a 256x256 image with two black squares on a white background.
###############################################
import cv2
import numpy as np

# Create a blank 256x256 image with a white background
blank_img = np.ones((256, 256, 3), dtype=np.uint8) * 255  # White background


# Draw the black squares
cv2.rectangle(blank_img, (128, 0), (256, 128), (0, 0, 0), -1)  # Top-right square
cv2.rectangle(blank_img, (0, 128), (128, 256), (0, 0, 0), -1)  # Bottom-left square

# Display the image in a window
cv2.imshow('', blank_img)  # Window title
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows
