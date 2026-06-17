import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Inisialisasi parameter top-level antarmuka Streamlit
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="wide")

# Terminasi elemen bawaan Streamlit (Menu, Footer) untuk optimalisasi viewport
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Render abstraksi header utama
st.title("🔍 Face Recognition & PCA Analysis Dashboard")
st.markdown("Sistem komparasi wajah dengan reduksi dimensi **PCA**, **Euclidean Distance**, dan **Cosine Similarity**.")

# Komputasi model PCA dan isolasi ke dalam memori cache
@st.cache_resource
def train_pca():
    with st.spinner("Menginisialisasi dataset dan melatih model PCA..."):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        # Resolusi komponen ditingkatkan untuk mereduksi hilangnya varians pada matriks
        pca = PCA(n_components=100, whiten=True)
        pca.fit(X)
        return pca, X

pca_model, dataset_matrix = train_pca()

# Instansiasi object deteksi wajah Viola-Jones (Haar Cascade)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    # Parsing byte stream image ke representasi array NumPy spasial
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Normalisasi histogram area lokal (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Reduksi high-frequency noise menggunakan kernel filter spasial
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Ekstraksi Region of Interest (ROI) wajah
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    
    # Modifikasi bounding box (Tighter Crop) untuk mengeliminasi distorsi background dan rambut
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    face_crop = img[y+margin_y : y+h-margin_y, x+margin_x : x+w-margin_x]
    
    # Downsampling resolusi spasial untuk sinkronisasi dengan dataset pre-trained
    face_resize = cv2.resize(face_crop, (64, 64))
    
    # Transformasi dimensi matriks dan skalar Min-Max scaling [0, 1]
    face_flat = face_resize.flatten() / 255.0 
    
    # Proyeksi vektor n-dimensi ke subspace PCA
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_resize

# Segmentasi layout UI via pembagian grid kolom
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Image Source 1")
    file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"], key="f1")

with col2:
    st.subheader("📷 Image Source 2")
    file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"], key="f2")

# Input parameter threshold via komponen antarmuka dinamis
st.markdown("---")
threshold = st.slider("⚙️ Threshold Toleransi Kesamaan (%)", min_value=60, max_value=95, value=75, step=1)

# Inisialisasi callback komputasi metrik
if st.button("⚖️ Run Analysis", type="primary", use_container_width=True):
    if file1 and file2:
        with st.spinner("Mengeksekusi proses komputasi vektor spasial..."):
            vec1, img_crop1 = process_face(file1)
            vec2, img_crop2 = process_face(file2)
            
            if vec1 is None or vec2 is None:
                # Terminasi eksekusi dengan exception handler
                st.error("Exception: Gagal mendeteksi matriks wajah. Pastikan input memiliki kontras spasial yang valid.")
            else:
                # Komputasi jarak metrik Euclidean (L2 Norm)
                diff = vec1 - vec2
                euclidean_dist = np.sqrt(np.sum(diff ** 2))
                
                # Komputasi rasio ortogonalitas (Cosine Similarity)
                dot_product = np.dot(vec1, vec2)
                norm_v1 = np.linalg.norm(vec1)
                norm_v2 = np.linalg.norm(vec2)
                cosine_sim = dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) != 0 else 0.0
                
                # Normalisasi koefisien [-1, 1] menjadi probabilitas persentase [0, 100]
                similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
                
                # Render klasifikasi akhir via conditional thresholding
                st.markdown("---")
                if similarity_percentage >= threshold:
                    st.success(f"### KESIMPULAN: WAJAH IDENTIK ({similarity_percentage}%)")
                else:
                    st.error(f"### KESIMPULAN: WAJAH BERBEDA ({similarity_percentage}%)")
                
                # Render struktur kontainer metric stream
                m1, m2, m3 = st.columns(3)
                m1.metric("Tingkat Kesamaan", f"{similarity_percentage}%")
                m2.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                m3.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
                
                # Render representasi dashboard matriks
                st.markdown("---")
                st.subheader("📊 Visualisasi Fitur Spasial PCA")
                
                tab1, tab2, tab3 = st.tabs(["Deteksi & Rekonstruksi", "Distribusi Vektor", "Statistik Matriks"])
                
                with tab1:
                    st.write("**1. Output Pre-processing ROI (CLAHE + Gaussian Blur)**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(img_crop1, caption="Input 1 (Processed)")
                    c2.image(img_crop2, caption="Input 2 (Processed)")
                    
                    # Dekompresi representasi fitur eigen ke topologi spasial 2D
                    rekon1 = pca_model.inverse_transform([vec1]).reshape(64, 64)
                    rekon2 = pca_model.inverse_transform([vec2]).reshape(64, 64)
                    
                    # Transformasi nilai tipe data floating point menjadi integer 8-bit untuk visualisasi citra
                    c3.image(np.clip(rekon1 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform 1")
                    c4.image(np.clip(rekon2 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform 2")

                with tab2:
                    st.write(f"**2. Proyeksi Spasial {pca_model.n_components} Principal Components**")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    x_axis = np.arange(pca_model.n_components)
                    ax.bar(x_axis - 0.2, vec1, 0.4, label='Vektor 1')
                    ax.bar(x_axis + 0.2, vec2, 0.4, label='Vektor 2')
                    
                    ax.set_xlabel("Indeks Komponen")
                    ax.set_ylabel("Magnitudo Koefisien")
                    ax.legend()
                    st.pyplot(fig)

                with tab3:
                    st.write("**3. Metrik Evaluasi Model PCA (Cumulative Ratio)**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Akumulasi distribusi varians spasial set pelatihan
                        explained_variance = pca_model.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        ax2.plot(cumulative_variance, marker='o', linestyle='-')
                        ax2.set_title("Cumulative Explained Variance")
                        ax2.set_xlabel("n_components")
                        ax2.set_ylabel("Rasio Varians")
                        ax2.grid(True)
                        st.pyplot(fig2)
                        st.caption(f"Retensi informasi spasial terakumulasi pada **{cumulative_variance[-1]*100:.1f}%**.")
                        
                    with col_b:
                        st.write("**Topologi Eigenface Utama (Index 0)**")
                        # Rekonstruksi struktur eigenvector dengan magnitudo tertinggi
                        eigenface_0 = pca_model.components_[0].reshape(64, 64)
                        fig3, ax3 = plt.subplots(figsize=(3, 3))
                        ax3.imshow(eigenface_0, cmap='gray')
                        ax3.axis('off')
                        st.pyplot(fig3)
    else:
        # Fallback eksekusi jika pre-kondisi file I/O tidak terpenuhi
        st.info("Sistem standby: Menunggu input file citra sebelum komputasi dieksekusi.")
