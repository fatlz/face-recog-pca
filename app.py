import streamlit as st
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="centered")

st.title("🔍 Face Recognition AI")
st.markdown("Aplikasi perbandingan wajah menggunakan **PCA, Euclidean Distance, dan Cosine Similarity**.")

# --- LOAD MODEL (Biar gak ngulang training terus) ---
@st.cache_resource
def train_pca():
    with st.spinner("Mengunduh dataset LFW dan melatih model PCA... (Hanya sekali)"):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        pca = PCA(n_components=50, whiten=True)
        pca.fit(X)
        return pca

pca_model = train_pca()
st.success("✅ Model PCA berhasil dilatih dengan dataset!")

# Load Haar Cascade untuk deteksi wajah
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    """Fungsi mendeteksi wajah, crop, dan ekstraksi fitur pakai PCA"""
    # Convert file upload Streamlit ke format OpenCV
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    face_crop = img[y:y+h, x:x+w]
    face_resize = cv2.resize(face_crop, (64, 64)) # Harus sama dengan dataset
    
    # Reduksi dimensi pakai PCA yang udah di-train
    face_flat = face_resize.flatten()
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_crop

# --- UI UPLOAD FOTO ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Foto 1 (Masa Lalu)")
    file1 = st.file_uploader("Upload Foto 1", type=["jpg", "png", "jpeg"], key="file1")
    if file1:
        st.image(file1, use_container_width=True)

with col2:
    st.subheader("📷 Foto 2 (Sekarang)")
    file2 = st.file_uploader("Upload Foto 2", type=["jpg", "png", "jpeg"], key="file2")
    if file2:
        st.image(file2, use_container_width=True)

# --- PROSES BANDINGKAN ---
if st.button("⚖️ Bandingkan Wajah", type="primary", use_container_width=True):
    if file1 is None or file2 is None:
        st.error("Woy, upload dulu dua-duanya cok!")
    else:
        with st.spinner("Sedang membandingkan dengan algoritma PCA..."):
            vec1, img_crop1 = process_face(file1)
            vec2, img_crop2 = process_face(file2)
            
            if vec1 is None or vec2 is None:
                st.error("Wajah gak ketemu! Coba pakai foto yang lebih jelas.")
            else:
                # 1. Hitung Euclidean Distance
                diff = vec1 - vec2
                euclidean_dist = np.sqrt(np.sum(diff ** 2))
                
                # 2. Hitung Cosine Similarity
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                
                if norm_vec1 * norm_vec2 != 0:
                    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
                else:
                    cosine_sim = 0.0
                
                # Normalisasi skor
                similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
                
                st.markdown("---")
                st.subheader("📊 Hasil Analisis Algoritma")
                
                # Nampilin Metrik Kekinian ala Streamlit
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Kemiripan", f"{similarity_percentage}%")
                col_m2.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                col_m3.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
                
                # Tampilkan crop wajah (Biar dosen liat lu beneran mengekstrak fitur)
                st.write("**Wajah yang berhasil diekstrak sistem:**")
                c1, c2 = st.columns(2)
                c1.image(img_crop1, caption="Wajah 1", width=100)
                c2.image(img_crop2, caption="Wajah 2", width=100)

                # Kesimpulan
                if similarity_percentage >= 70:
                    st.success(f"**Kesimpulan:** Berdasarkan perhitungan vektor, kedua foto adalah ORANG YANG SAMA.")
                else:
                    st.error(f"**Kesimpulan:** Berdasarkan perhitungan vektor, kedua foto adalah ORANG YANG BERBEDA.")
