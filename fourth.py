import keras_ocr
import matplotlib.pyplot as plt

# Step 1: Build the pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Step 2: Read the te
image_path = 'test2.png'  # <--- Change to your file path
image = keras_ocr.tools.read(image_path)

# Step 3: Run pipeline (detection + recognition)
prediction_groups = pipeline.recognize([image])

# Step 4: Display the results
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)

# Step 5: Print text predictions
print("\nDetected texts:")
for text, box in prediction_groups[0]:
    print(f"Detected text: '{text}'")