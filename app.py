import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import time

# Konfigurasi top-level antarmuka dan metadata halaman
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="wide")

# Terminasi elemen margin atas dan menu default Streamlit untuk optimalisasi viewport
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Render header viewport utama
st.title("🔍 Face Recognition & PCA Dashboard")
st.markdown("Sistem komparasi wajah spasial berbasis reduksi dimensi dimensional.")

# Panel informasi eksplisit untuk validasi akademis (Requirement Dosen)
st.info("""
**Spesifikasi Arsitektur Sistem:**
* **Metode Ekstraksi Fitur:** Principal Component Analysis (PCA)
* **Dataset Pelatihan (Pre-trained):** Olivetti Faces Database
* **Metrik Komparasi Vektor 1:** Euclidean Distance (L2 Norm)
* **Metrik Komparasi Vektor 2:** Cosine Similarity (Dot Product)
""")
st.divider()

# Alokasi cache memori untuk pre-trained model PCA
@st.cache_resource
def train_pca():
    with st.spinner("Menginisialisasi dataset latih dan matriks PCA..."):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        # Penetapan 100 komponen utama untuk retensi varians spasial
        pca = PCA(n_components=100, whiten=True)
        pca.fit(X)
        return pca, X

pca_model, dataset_matrix = train_pca()

# Instansiasi objek deteksi Viola-Jones (Haar Cascade)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    # Parsing input memori ke representasi array spasial
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Pre-processing: Histogram equalization (CLAHE) dan spatial smoothing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Ekstraksi Region of Interest (ROI)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    
    # Modifikasi bounding box (Tighter Crop) untuk eliminasi noise periferal
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)
    face_crop = img[y+margin_y : y+h-margin_y, x+margin_x : x+w-margin_x]
    
    # Downsampling resolusi dan min-max scaling vektor [0, 1]
    face_resize = cv2.resize(face_crop, (64, 64))
    face_flat = face_resize.flatten() / 255.0 
    
    # Proyeksi vektor n-dimensi ke subspace PCA
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_resize

# Parameter threshold dieksekusi di blok tata letak utama (Main Layout)
st.markdown("### ⚙️ Parameter Kontrol Toleransi")
threshold = st.slider("Penyesuaian Sensitivitas Probabilitas Kesamaan (%)", min_value=60, max_value=95, value=75, step=1)
st.markdown("---")

# Fragmentasi layout ke dalam struktur grid 2 kolom untuk I/O
col1, col2 = st.columns(2)
with col1:
    st.info("Unggah citra referensi (Input A)")
    file1 = st.file_uploader("Upload Foto 1", type=["jpg", "png", "jpeg"], key="f1", label_visibility="collapsed")

with col2:
    st.info("Unggah citra target (Input B)")
    file2 = st.file_uploader("Upload Foto 2", type=["jpg", "png", "jpeg"], key="f2", label_visibility="collapsed")

# Inisialisasi event listener komputasi
if st.button("🚀 Eksekusi Komputasi Matriks", type="primary", use_container_width=True):
    if file1 and file2:
        # Rendering komponen progress bar dinamis
        progress_bar = st.progress(0, text="Menginisialisasi ekstraksi fitur...")
        
        vec1, img_crop1 = process_face(file1)
        progress_bar.progress(40, text="Memproses transformasi matriks spasial citra A...")
        time.sleep(0.3) 
        
        vec2, img_crop2 = process_face(file2)
        progress_bar.progress(80, text="Memproses transformasi matriks spasial citra B...")
        time.sleep(0.3)
        
        if vec1 is None or vec2 is None:
            progress_bar.empty()
            st.error("Exception: Gagal mendeteksi topologi wajah. Evaluasi kontras spasial citra input.")
        else:
            # Komputasi distance metrics spasial
            diff = vec1 - vec2
            euclidean_dist = np.sqrt(np.sum(diff ** 2))
            
            dot_product = np.dot(vec1, vec2)
            norm_v1 = np.linalg.norm(vec1)
            norm_v2 = np.linalg.norm(vec2)
            cosine_sim = dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) != 0 else 0.0
            
            similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
            
            progress_bar.progress(100, text="Komputasi selesai.")
            time.sleep(0.3)
            progress_bar.empty()
            
            # Klasifikasi logical thresholding
            st.write("### Hasil Evaluasi Vektor")
            if similarity_percentage >= threshold:
                st.success(f"**IDENTIK** — Tingkat kesamaan spasial: {similarity_percentage}%")
            else:
                st.error(f"**BERBEDA** — Tingkat kesamaan spasial: {similarity_percentage}%")
            
            # Rendering struktur metrik kuantitatif
            m1, m2, m3 = st.columns(3)
            m1.metric("Kemiripan Cosine", f"{similarity_percentage}%")
            m2.metric("Nilai Cosine Similarity", f"{cosine_sim:.4f}")
            m3.metric("Euclidean Distance (L2)", f"{euclidean_dist:.4f}")
            
            # Render blok tabulasi analitik secara statis (Tanpa Expander)
            st.markdown("---")
            st.subheader("📊 Detail Analisis Matriks & Visualisasi Spasial PCA")
            tab1, tab2, tab3 = st.tabs(["Rekonstruksi Spasial", "Distribusi Fitur Vektor", "Evaluasi Model"])
            
            with tab1:
                st.write("**Output Isolasi ROI (CLAHE + Filtering)**")
                c1, c2, c3, c4 = st.columns(4)
                c1.image(img_crop1, caption="Input A (Processed)")
                c2.image(img_crop2, caption="Input B (Processed)")
                
                # Dekompresi vektor eigen ke arsitektur spasial 2D
                rekon1 = pca_model.inverse_transform([vec1]).reshape(64, 64)
                rekon2 = pca_model.inverse_transform([vec2]).reshape(64, 64)
                
                c3.image(np.clip(rekon1 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform A")
                c4.image(np.clip(rekon2 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform B")

            with tab2:
                st.write(f"**Proyeksi Vektor pada {pca_model.n_components} Principal Components**")
                fig, ax = plt.subplots(figsize=(10, 3))
                
                x_axis = np.arange(pca_model.n_components)
                ax.bar(x_axis - 0.2, vec1, 0.4, label='Vektor A')
                ax.bar(x_axis + 0.2, vec2, 0.4, label='Vektor B')
                
                ax.set_xlabel("Indeks Komponen Dimensional")
                ax.set_ylabel("Magnitudo Koefisien")
                ax.legend()
                st.pyplot(fig)

            with tab3:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Cumulative Explained Variance**")
                    explained_variance = pca_model.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.plot(cumulative_variance, marker='o', linestyle='-')
                    ax2.set_xlabel("Jumlah Komponen Utama")
                    ax2.set_ylabel("Rasio Varians Terenkapsulasi")
                    ax2.grid(True)
                    st.pyplot(fig2)
                    st.caption(f"Retensi informasi terakumulasi: **{cumulative_variance[-1]*100:.1f}%**.")
                    
                with col_b:
                    st.write("**Topologi Eigenface (Index 0)**")
                    # Rendering struktur eigenvector dengan varians absolut tertinggi
                    eigenface_0 = pca_model.components_[0].reshape(64, 64)
                    fig3, ax3 = plt.subplots(figsize=(3, 3))
                    ax3.imshow(eigenface_0, cmap='gray')
                    ax3.axis('off')
                    st.pyplot(fig3)
    else:
        st.warning("Pre-condition failed: Masukkan kedua file citra pada antarmuka I/O di atas.")
