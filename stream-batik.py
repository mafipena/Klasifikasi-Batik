import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model('C:\\Batik\\MobileNet-V2-Batik bismillah-83.99.h5')

# Daftar nama batik yang sesuai dengan kelas
batik_names = ['Batik Bali', 'Batik Betawi', 'Batik Celup', 'Batik Cendrawasih',
               'Batik Dayak', 'Batik Kawung', 'Batik Lasem', 'Batik Megamendung',
               'Batik Parang', 'Batik Tambal']

# Fungsi untuk memuat dan memproses gambar
def load_and_preprocess_image(image):
    # Mengkonversi gambar ke RGB
    img = image.convert('RGB')

    # Mengubah ukuran gambar agar sesuai dengan ukuran input model (224x224)
    img = img.resize((224, 224))

    # Menormalkan nilai piksel ke rentang [0, 1]
    img = np.array(img) / 255.0

    # Menambahkan dimensi batch
    img = np.expand_dims(img, axis=0)

    return img

# Fungsi untuk melakukan prediksi
def predict_image(model, image):
    # Melakukan prediksi
    predictions = model.predict(image)

    # Mendapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class

# Aplikasi Streamlit
st.title('Klasifikasi Gambar Batik')
st.write("""
### Selamat datang di aplikasi klasifikasi gambar batik!
Silakan unggah gambar batik yang ingin Anda klasifikasikan.
""")

# Menggunakan sidebar untuk file uploader
st.sidebar.title("Unggah Gambar Batik")
uploaded_file = st.sidebar.file_uploader("Pilih gambar batik...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Memuat dan memproses gambar
    image = Image.open(uploaded_file)
    processed_image = load_and_preprocess_image(image)

    # Melakukan prediksi
    predicted_class = predict_image(model, processed_image)

    # Menampilkan gambar dan hasil prediksi
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
    st.write(f"**Batik yang Diprediksi: {batik_names[predicted_class[0]]}**")

    st.success(f"Batik yang Diprediksi: **{batik_names[predicted_class[0]]}**")
else:
    st.sidebar.write("Silakan unggah gambar untuk klasifikasi.")
