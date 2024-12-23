import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images with a title
def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load satellite image
image_path = '/content/satellite_image.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}. Check the file path.")
    exit()

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(gray, "Grayscale Image")

# Step 2: Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
show_image(blurred, "Gaussian Blurred Image")

# Step 3: Use Otsu's thresholding method to automatically choose a threshold value
ret, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show_image(thresholded, "Otsu's Thresholded Image")

# Step 4: Apply morphological operations to refine the road extraction
kernel = np.ones((3, 3), np.uint8)

# Erosion
eroded = cv2.erode(thresholded, kernel, iterations=1)
show_image(eroded, "Eroded Image")

# Dilation
dilated = cv2.dilate(eroded, kernel, iterations=1)
show_image(dilated, "Dilated Image")



# Optional: Overlay the extracted roads on the original image
overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR), 0.3, 0)
show_image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "Overlayed Roads")

# Save the final processed image
output_path = 'extracted_roads.jpg'
cv2.imwrite(output_path, eroded)
print(f"Processed image saved at {output_path}")

# Final step: Show the result after all operations
plt.imshow(eroded, cmap='gray')
plt.title("Final Extracted Roads")
plt.axis('off')
plt.show()
