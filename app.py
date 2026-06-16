import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="wide")

st.title("🔍 Face Recognition & PCA Analysis Dashboard")
st.markdown("Sistem komparasi wajah dengan reduksi dimensi **PCA**, **Euclidean Distance**, dan **Cosine Similarity**.")

# --- LOAD MODEL & DATASET ---
@st.cache_resource
def train_pca():
    with st.spinner("Mengunduh dataset LFW dan melatih model PCA... (Hanya sekali)"):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        pca = PCA(n_components=50, whiten=True)
        pca.fit(X)
        return pca, X

pca_model, dataset_matrix = train_pca()
st.success("✅ Model PCA berhasil dilatih dengan dataset Olivetti Faces!")

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    face_crop = img[y:y+h, x:x+w]
    face_resize = cv2.resize(face_crop, (64, 64))
    
    face_flat = face_resize.flatten()
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_resize

# --- UI UPLOAD FOTO ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Foto 1 (Masa Lalu)")
    file1 = st.file_uploader("Upload Foto 1", type=["jpg", "png", "jpeg"], key="f1")

with col2:
    st.subheader("📷 Foto 2 (Sekarang)")
    file2 = st.file_uploader("Upload Foto 2", type=["jpg", "png", "jpeg"], key="f2")

# --- PROSES BANDINGKAN ---
if st.button("⚖️ Bandingkan dan Analisis", type="primary", use_container_width=True):
    if file1 and file2:
        with st.spinner("Sedang mengekstrak fitur matematis PCA..."):
            vec1, img_crop1 = process_face(file1)
            vec2, img_crop2 = process_face(file2)
            
            if vec1 is None or vec2 is None:
                st.error("Wajah gak ketemu! Coba pakai foto yang pencahayaannya lebih jelas.")
            else:
                # Perhitungan Jarak
                diff = vec1 - vec2
                euclidean_dist = np.sqrt(np.sum(diff ** 2))
                
                dot_product = np.dot(vec1, vec2)
                norm_v1 = np.linalg.norm(vec1)
                norm_v2 = np.linalg.norm(vec2)
                cosine_sim = dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) != 0 else 0.0
                similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
                
                # --- HASIL UTAMA ---
                st.markdown("---")
                if similarity_percentage >= 70:
                    st.success(f"### KESIMPULAN: WAJAH COCOK ({similarity_percentage}%) - ORANG YANG SAMA")
                else:
                    st.error(f"### KESIMPULAN: WAJAH BERBEDA ({similarity_percentage}%)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Kemiripan", f"{similarity_percentage}%")
                m2.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                m3.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
                
                # --- SECTION ANALISIS VISUAL (DASHBOARD) ---
                st.markdown("---")
                st.subheader("📊 Analisis Visual PCA (Principal Component Analysis)")
                
                tab1, tab2, tab3 = st.tabs(["Deteksi & Rekonstruksi", "Distribusi Vektor", "Statistik PCA"])
                
                with tab1:
                    st.write("**1. Hasil Cropping OpenCV (64x64 pixel)**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(img_crop1, caption="Wajah 1 (Asli)")
                    c2.image(img_crop2, caption="Wajah 2 (Asli)")
                    
                    # Rekonstruksi balik dari PCA ke Gambar (Biar keliatan kerjanya PCA)
                    rekon1 = pca_model.inverse_transform(vec1).reshape(64, 64)
                    rekon2 = pca_model.inverse_transform(vec2).reshape(64, 64)
                    
                    c3.image(np.clip(rekon1, 0, 255).astype(np.uint8), caption="Rekonstruksi PCA 1")
                    c4.image(np.clip(rekon2, 0, 255).astype(np.uint8), caption="Rekonstruksi PCA 2")

                with tab2:
                    st.write("**2. Perbandingan 50 Fitur Komponen (Vektor)**")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    x_axis = np.arange(50)
                    ax.bar(x_axis - 0.2, vec1, 0.4, label='Wajah 1', color='blue')
                    ax.bar(x_axis + 0.2, vec2, 0.4, label='Wajah 2', color='orange')
                    ax.set_xlabel("Principal Component ke-n")
                    ax.set_ylabel("Nilai Bobot")
                    ax.legend()
                    st.pyplot(fig)
                    st.caption("Jika diagram batang biru dan oranye polanya mirip, nilai Euclidean akan kecil dan Cosine akan mendekati 1.")

                with tab3:
                    st.write("**3. Evaluasi Kinerja PCA (Dataset Level)**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Grafik Cumulative Variance
                        explained_variance = pca_model.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        ax2.plot(cumulative_variance, marker='o', linestyle='-', color='purple')
                        ax2.set_title("Akumulasi Informasi PCA (Cumulative Variance)")
                        ax2.set_xlabel("Jumlah Komponen")
                        ax2.set_ylabel("Varians Terjelaskan")
                        ax2.grid(True)
                        st.pyplot(fig2)
                        st.caption(f"50 komponen utama mampu mempertahankan **{cumulative_variance[-1]*100:.1f}%** informasi dari gambar asli.")
                        
                    with col_b:
                        # Nampilin "Eigenface" pertama (Wajah Rata-rata dari Dataset)
                        st.write("**Eigenface Utama (Komponen #1)**")
                        eigenface_0 = pca_model.components_[0].reshape(64, 64)
                        fig3, ax3 = plt.subplots(figsize=(3, 3))
                        ax3.imshow(eigenface_0, cmap='gray')
                        ax3.axis('off')
                        st.pyplot(fig3)
                        st.caption("Ini adalah pola fitur wajah paling dominan yang diekstrak algoritma dari dataset latih.")
    else:
        st.warning("Silakan unggah kedua foto terlebih dahulu.")
