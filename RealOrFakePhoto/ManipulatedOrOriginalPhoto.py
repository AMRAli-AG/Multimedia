"""
Image Manipulation Detection Based on Histogram Spikes and Gaps

Author: Amr Ali 
Date: 19/11/2024

Description:
This script detects potential image manipulations by analyzing spikes and gaps in the histograms of two images.
The code reads two images, computes their histograms, and calculates the difference between consecutive histogram values. 
Significant spikes (abrupt increases) and gaps (sudden drops) in the histogram are identified and used to
classify the images as either "Original" or "Manipulated" based on a defined threshold.

Key Functions:
1. detect_spikes_and_gaps(image_path, threshold=50): 
   - This function reads an image, computes its grayscale histogram, and identifies spikes and
     gaps in the histogram based on a specified threshold.

2. plot_results(image_path1, spikes1, gaps1, hist1, diffs1, image_path2, spikes2, gaps2, hist2, diffs2, result1, result2): 
   - This function visualizes the original images, their histograms, and the detected spikes and gaps
     for both images. The images are displayed with their respective manipulation analysis.

3. compare_images(image_path1, image_path2, threshold=100): 
   - This function compares two images by detecting spikes and gaps in their histograms.
     It prints the results and classifies each image as "Manipulated" or "Original" based on the threshold.

Input:
- Two image paths (image_path1 and image_path2) to be compared.

Output:
- The script outputs the number of spikes and gaps for both images and classifies the images as "Manipulated" or "Real" based on the threshold.
- It also generates a plot showing the images along with their histograms and marked spikes and gaps.

Usage:
1. Replace the image paths with the paths of the images you want to analyze.
2. Adjust the threshold value if needed to refine the detection of manipulations.
3. Run the script to visualize the images and analyze their authenticity.

Example:
    image_path1 = r'C:\path\to\image1.png'
    image_path2 = r'C:\path\to\image2.png'
    compare_images(image_path1, image_path2, threshold=80)

Notes:
- The images should be of similar nature and resolution for meaningful comparison.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Spikes and Gaps in Histogram
def detect_spikes_and_gaps(image_path, threshold=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    diffs = np.diff(hist)

    spikes = np.where(diffs > threshold)[0]  # Positions where the difference is high (spikes)
    gaps = np.where(diffs < -threshold)[0]    # Positions where the difference is low (gaps)

    return len(spikes), len(gaps), hist, diffs

# Plotting histogram, spikes/gaps, and images
def plot_results(image_path1, spikes1, gaps1, hist1, diffs1, image_path2, spikes2, gaps2, hist2, diffs2, result1, result2):
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot for Image 1: Image display
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f"Image 1\n{result1}")

    # Plot for Image 2: Image display
    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(f"Image 2\n{result2}")

    # Plot for Image 1: Histogram Difference with Spikes and Gaps
    axes[1, 0].bar(range(len(diffs1)), diffs1, color='gray', label='Histogram Difference')
    axes[1, 0].scatter(spikes1, diffs1[spikes1], color='red', label='Spikes')
    axes[1, 0].scatter(gaps1, diffs1[gaps1], color='blue', label='Gaps')
    axes[1, 0].set_title('Image 1 Differences with Spikes and Gaps')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Difference')
    axes[1, 0].legend()

    # Plot for Image 2: Histogram Difference with Spikes and Gaps
    axes[1, 1].bar(range(len(diffs2)), diffs2, color='gray', label='Histogram Difference')
    axes[1, 1].scatter(spikes2, diffs2[spikes2], color='red', label='Spikes')
    axes[1, 1].scatter(gaps2, diffs2[gaps2], color='blue', label='Gaps')
    axes[1, 1].set_title('Image 2 Differences with Spikes and Gaps')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Difference')
    axes[1, 1].legend()

    # Adjust layout to prevent overlapping subplots
    plt.tight_layout()
    plt.show()

# Compare two images based on spikes and gaps
def compare_images(image_path1, image_path2, threshold=100):
    # Detect spikes and gaps for the first image
    spikes1, gaps1, hist1, diffs1 = detect_spikes_and_gaps(image_path1)
    
    # Detect spikes and gaps for the second image
    spikes2, gaps2, hist2, diffs2 = detect_spikes_and_gaps(image_path2)

    # Print the results for both images
    print(f"Image 1: Spikes: {spikes1}, Gaps: {gaps1}")
    print(f"Image 2: Spikes: {spikes2}, Gaps: {gaps2}")
    
    # Compare and determine authenticity based on spikes and gaps
    if spikes1 > threshold and gaps1 > threshold:
        result1 = "Image 1 is Manipulated"
    else:
        result1 = "Image 1 isreal"
    
    if spikes2 > threshold and gaps2 > threshold:
        result2 = "Image 2 is Manipulated"
    else:
        result2 = "Image 2 is likely real"

    # Print results
    print(result1)
    print(result2)

    # Plot the results for both images in one window with images displayed
    plot_results(image_path1, spikes1, gaps1, hist1, diffs1, image_path2, spikes2, gaps2, hist2, diffs2, result1, result2)

# Example usage with image paths
image_path1 = r'C:\Users\amr15\Pictures\Screenshots\Screenshot 2024-11-16 004135.png'  # Replace with the path to the first image
image_path2 = r'C:\Users\amr15\Desktop\Screenshot 2024-11-16 004135.png'  # Replace with the path to the second image

compare_images(image_path1, image_path2, threshold=80)  # Adjust the threshold as needed
