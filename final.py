import os
import requests
import keras_ocr
import matplotlib.pyplot as plt
import pandas as pd
from fuzzywuzzy import fuzz, process

def run_ocr_and_match(image_path, csv_path, match_column='name', limit=5):
    # Build the OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Read and OCR the image
    image = keras_ocr.tools.read(image_path)
    predictions = pipeline.recognize([image])[0]

    # Show annotated image
    fig, ax = plt.subplots(figsize=(10, 10))
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    plt.show()

    # Collect and print detected text
    ocr_texts = [text for text, _ in predictions if text.strip()]
    print("\n📝 Detected texts:")
    for text in ocr_texts:
        print(f"- {text}")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Build a single query string
    query = ' '.join(ocr_texts)

    # Fuzzy-match against the 'name' column
    choices = df[match_column].dropna().tolist()
    matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)

    # Collect the matched rows
    matched_rows = []
    print("\n🔍 Top OCR Matches:\n" + "-"*40)
    for i, (name, score) in enumerate(matches, 1):
        if score < 54:
            print(f"\n{i}. ❌ Rejected Match: '{name}'  |  Score: {score} (Below threshold)")
            continue
        row = df[df[match_column] == name].iloc[0]
        matched_rows.append(row)
        print(f"\n{i}. ✅ Match: '{name}'  |  Score: {score}")
        print(row.to_frame().T)

    return matched_rows

def download_images_for_rows(rows, url_column='image_url', save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    print("\n⬇️ Downloading images for matched rows:")
    for row in rows:
        url = row.get(url_column, '')
        if not isinstance(url, str) or not url.startswith('http'):
            print(f"⚠️  No valid URL for ID {row['id']}, skipping.")
            continue
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"❌  Failed to download ID {row['id']} ({url}): {e}")
            continue
        filename = f"{row['id']}.jpg"
        path = os.path.join(save_dir, filename)
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"✅  Saved image for ID {row['id']} → {path}")

if __name__ == "__main__":
    image_file = "test6.png"
    dataset_file = "dataset.csv"

    # 1) OCR + match
    matched = run_ocr_and_match(image_file, dataset_file)

    # 2) Download only those matched
    download_images_for_rows(matched)
