###############################################
# Author: Amr Ali 
# Date: 18/10
# Version: 1.0
###############################################
# Description: 
# This script reads a color image, creates three copies of it, and manipulates each copy 
# to display fully saturated versions of the green, red, and blue channels respectively. 
# The original image and the three modified images are then displayed side by side 
# in a 2x2 grid using Matplotlib.
#
# Usage: 
# - Ensure that the required image file is available in the specified path.
# - Run the script using a Python environment with OpenCV and Matplotlib installed.
#
# Dependencies: 
# - Python 3.x
# - OpenCV (cv2)
# - Matplotlib (for displaying images)
#
# Input: 
# - A color image file (e.g., 'ceramic-pro-007-768x512.jpg')
#
# Output: 
# - Displays the original image and three versions of the image with saturated green, red, 
#   and blue channels in a 2x2 grid.
###############################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the original image and create three copies for channels 
img_real = cv2.imread(r'C:\Users\amr15\cvc\ceramic-pro-007-768x512.jpg',1)
img_green = cv2.imread(r'C:\Users\amr15\cvc\ceramic-pro-007-768x512.jpg',1)
img_red = cv2.imread(r'C:\Users\amr15\cvc\ceramic-pro-007-768x512.jpg',1)
img_blue = cv2.imread(r'C:\Users\amr15\cvc\ceramic-pro-007-768x512.jpg',1)

img_green[:,:,1]=255# Modify the green channel by setting all green values to 255
img_red[:,:,0]=255# Modify the red channel by setting all red values to 255
img_blue[:,:,-1]=255# Modify the blue channel by setting all blue values to 255

plt.subplot(2,2,1) # Create a 2x2 subplot to display the images In the first quarter
plt.imshow(img_real) # Display the real Image
plt.title('real Image') # Title for the plot
plt.axis('off')  # Hide axes


plt.subplot(2,2,2) # Create a 2x2 subplot to display the images In the second quarter 
plt.imshow(img_green) # Display the Green channel
plt.title('Green channel') # Title for the plot
plt.axis('off')  # Hide axes

plt.subplot(2,2,3)  # Create a 2x2 subplot to display the images In the third quarter
plt.imshow(img_red) # Display the red channel
plt.title('red channel') # Title for the plot
plt.axis('off')  # Hide axes

plt.subplot(2,2,4)  # Create a 2x2 subplot to display the images In the fourth quarter
plt.imshow(img_blue) # Display the blue channel
plt.title('blue channel') # Title for the plot
plt.axis('off')  # Hide axes

plt.show()# Display the figure with all subplots


cv2.waitKey(0)
cv2.destroyAllWindows()

