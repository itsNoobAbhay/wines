import cv2
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np

# 1. Read image
image_path = 'test2.png'
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError("Image not found at given path!")

# 2. Convert to grayscale
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 3. Histogram Equalization
equalized_gray = cv2.equalizeHist(gray)

# 4. Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_img = clahe.apply(equalized_gray)

# 5. Sharpen the image
# Sharpening kernel
sharpen_kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
sharpened_img = cv2.filter2D(contrast_img, -1, sharpen_kernel)

# 6. Convert back to 3-channel (for OCR compatibility)
prepped_image = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2RGB)

# 7. Run OCR
pipeline = keras_ocr.pipeline.Pipeline()
prediction_groups = pipeline.recognize([prepped_image])
predictions = prediction_groups[0]

# 8. Visualize the results
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))  # Show original image for context
keras_ocr.tools.drawAnnotations(image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), predictions=predictions, ax=ax)
plt.axis('off')
plt.show()

# 9. Print detected text
print("\nDetected Texts:")
for text, box in predictions:
    print(f"• {text}")
