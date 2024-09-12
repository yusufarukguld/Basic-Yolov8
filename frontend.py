import streamlit as st
import torch
from PIL import Image
import io

# Modeli yükleyin
model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)

st.title('Fotoğraf ile Nesne Tespiti')

uploaded_file = st.file_uploader("Fotoğraf yükleyin", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))
    results = model(image)
    
    # Tespit sonuçlarını çiz ve göster
    results.render()
    for img in results.imgs:
        img_byte_arr = io.BytesIO()
        img_pil = Image.fromarray(img)
        img_pil.save(img_byte_arr, format='JPEG')
        st.image(img_byte_arr.getvalue())

st.write("Yüklenen fotoğrafta tespit edilen nesneler yukarıda gösterilmiştir.")
