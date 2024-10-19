###############################################
# Author: Amr Ali 
# Date: 19/10
# Version: 1.0
###############################################
# Description: 
# This script reads a color image, splits it into its red, green, and blue (RGB) channels, 
# swaps the red and blue channels, and displays the original and modified images side by side.
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
# - A color image file (e.g., 'image.jpg')
#
# Output: 
# - Displays the original image and the modified image with swapped red and blue channels.
###############################################

import cv2
import matplotlib.pyplot as plt


img = cv2.imread(r'C:\Users\amr15\cvc\gettyimages-1143289490-1024x1024.jpg',1)# Load the original image
B,R,G =cv2.split(img)# Split the image into B, G, R channels

swapped_img = cv2.merge([G,R, B])# Swap the red and blue channels

plt.subplot(1,2,1)  # Create a 2x2 subplot to display the images In the third quarter
plt.imshow(swapped_img) # Display the swapped img
plt.title('RGB channels') # Title for the plot
plt.axis('off')

plt.subplot(1,2,2) # Create a 2x2 subplot to display the images In the second quarter 
plt.imshow(img) # Display the R&B swapped image
plt.title('R&B swapped image') # Title for the plot
plt.axis('off')  # Hide axes

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
