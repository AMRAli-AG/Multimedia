###############################################
# Author: Amr Ali 
# Date: 27/10
# Version: 1.0
###############################################
# Description: 
# This script reads a grayscale image and modifies each pixel's intensity based on certain 
# threshold conditions. Specifically, pixels with values greater than 8 are set to 255 (white), 
# while those with values less than or equal to 8 are set to 0 (black). The modified grayscale 
# image is displayed using OpenCV.
# Usage: 
# - Ensure the required image files are available at the specified paths.
# - Run the script in a Python environment with OpenCV and .
#
# Dependencies: 
# - Python 3.x
# - OpenCV (cv2)
#
# Input: 
# - A grayscale image file (e.g., '3dccbc8e-59d2-43c3-8cb1-816ba35b9b54.png')
#
# Output: 
# - Displays the modified grayscale image with pixel intensity changes based on threshold conditions.
###############################################

import cv2

image = cv2.imread(r'3dccbc8e-59d2-43c3-8cb1-816ba35b9b54.png', cv2.IMREAD_GRAYSCALE)


    # Get the dimensions of the image
height, width = image.shape

    # Set the loop range based on the desired dimensions (up to 471x355 or the actual image size, whichever is smaller)
max_x = min(width, 471)
max_y = min(height, 355)

    # Loop over each pixel within the specified range
for y in range(max_y):
        for x in range(max_x):
            pixel_value = image[y, x]
            
            # Check pixel value and modify it based on the condition
            if pixel_value > 8:
                image[y, x] = 255
            elif pixel_value < 8 :
                image[y, x] = 0
cv2.imshow(r'C:\Users\amr15\cvc\modified_image.png', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()
print("Image modified and saved successfully.")
