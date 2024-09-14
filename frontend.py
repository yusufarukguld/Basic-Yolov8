import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def app():
    st.header('Nesne Tespit Uygulaması')
    st.subheader('Yüklenen resimdeki nesneleri tespit eder')
    st.write('Hoşgeldiniz, bu uygulama yüklediğiniz resimdeki nesneleri tespit eder. Sağladığımız hizmetin en basit gösterimidir ve bu alanda çok daha kapsamlı çözümler sunmaktayız.')
    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Fotoğraf yükle", type=['jpg', 'png'])
        selected_objects = st.multiselect('Tespit etmek istediğin nesneleri seç', object_names, default=['person'])
        min_confidence = st.slider('Güven Değeri', 0.0, 1.0)
        st.form_submit_button(label='Tespit Et')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        draw = ImageDraw.Draw(image)

        with st.spinner('Fotoğraf işleniyor...'):
            result = model(image)

            for detection in result[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name = model.names[cls]
                label = f'{object_name} {score}'

                if model.names[cls] in selected_objects and score > min_confidence:
                    draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
                    font = ImageFont.load_default()
                    draw.text((x0, y0 - 10), label, fill="blue", font=font)

            st.image(image, use_column_width=True)

if __name__ == "__main__":
    app()
