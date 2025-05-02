import os
import requests
import keras_ocr
import pandas as pd
from fuzzywuzzy import fuzz, process
import streamlit as st
from PIL import Image
import tempfile


DEMO_IMAGES = {
    "Select a demo": None,
    "Demo Wine 1": "demo/demo1.png",
    "Demo Wine 2": "demo/demo2.png",
    "Demo Wine 3": "demo/demo3.png",
    "Demo Wine 4": "demo/demo4.png",
    
    # Add more as needed
}



@st.cache_resource
def load_pipeline():
    return keras_ocr.pipeline.Pipeline()

@st.cache_data
def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def run_ocr_and_match(image, df, match_column='name', limit=5, score_threshold=54):
    pipeline = load_pipeline()
    img = keras_ocr.tools.read(image)
    predictions = pipeline.recognize([img])[0]

    ocr_texts = [text for text, _ in predictions if text.strip()]
    query = ' '.join(ocr_texts)
    choices = df[match_column].dropna().tolist()
    matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)

    matched_rows = []
    rejected_matches = []
    for name, score in matches:
        if score < score_threshold:
            rejected_matches.append((name, score))
            continue
        row = df[df[match_column] == name].iloc[0]
        matched_rows.append((row, score))
    return matched_rows, ocr_texts, rejected_matches

def download_images_for_rows(rows, url_column='image_url', save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    for row, _ in rows:
        url = row.get(url_column, '')
        if not isinstance(url, str) or not url.startswith('http'):
            continue
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            filename = f"{row['id']}.jpg"
            path = os.path.join(save_dir, filename)
            with open(path, 'wb') as f:
                f.write(r.content)
        except:
            pass

def display_card(row, score):
    st.markdown("----")
    st.markdown(f"### ✅ Match: `{row['name']}` (Score: **{score}**)")
    cols = st.columns(2)
    with cols[0]:
        for col in row.index:
            if col != 'image_url':
                st.write(f"**{col}:** {row[col]}")
    with cols[1]:
        if 'image_url' in row and isinstance(row['image_url'], str) and row['image_url'].startswith('http'):
            st.image(row['image_url'], width=250, caption="Matched Image")
            
# Streamlit app UI
st.set_page_config(page_title="Wine OCR Matcher", layout="wide")
st.title("🍷 Baxus Goggles")

dataset_file = "dataset.csv"

# Sidebar for demo image selection
st.sidebar.title("🖼️ Try a Demo Image")
selected_demo = st.sidebar.selectbox("Choose a demo image", list(DEMO_IMAGES.keys()))
demo_path = DEMO_IMAGES.get(selected_demo)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a wine label image", type=["png", "jpg", "jpeg"])

# Determine which image to use
image_path = None
image_caption = ""

if demo_path:
    image_path = demo_path
    image_caption = f"Demo: {selected_demo}"
elif uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name
    image_caption = "Uploaded Image"

# Only proceed if an image is available
if image_path:
    st.image(image_path, caption=image_caption, width=400)
    df = load_dataset(dataset_file)

    with st.spinner("🔍 Running OCR and matching..."):
        matched_rows, ocr_texts, rejected = run_ocr_and_match(image_path, df)

    st.subheader("🔍 Matches Found (Score ≥ 54)")
    if matched_rows:
        for row, score in matched_rows:
            display_card(row, score)
    else:
        st.warning("No matches with score ≥ 54 found.")

    if rejected:
        with st.expander("❌ Rejected Matches (Score < 54)"):
            for name, score in rejected:
                st.write(f"- {name}: Score {score}")

    if st.button("📥 Download matched images"):
        download_images_for_rows(matched_rows)
        st.success("Images downloaded to `images/` folder.")
else:
    st.info("Please upload an image or select a demo image from the sidebar.")
