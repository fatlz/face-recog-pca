import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Inisialisasi parameter Streamlit UI
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="wide")

# Injeksi CSS untuk overide properti komponen default Streamlit
custom_css = """
<style>
    /* Root application background gradient */
    .stApp {
        background: linear-gradient(135deg, #10223f, #051024);
        color: #ccd6f6;
    }
    
    /* Global text color override */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ccd6f6 !important;
    }
    
    /* Primary button styling and hover states */
    div.stButton > button:first-child {
        background-color: transparent !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: rgba(100, 255, 218, 0.15) !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.3) !important;
        transform: translateY(-2px);
    }
    
    /* Metric container styling dengan efek backdrop-filter */
    div[data-testid="metric-container"] {
        background: rgba(5, 16, 36, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Override warna font pada komponen metric */
    div[data-testid="stMetricValue"] {
        color: #64ffda !important;
    }
    
    /* Styling state pada komponen Tab */
    button[data-baseweb="tab"] {
        color: #8892b0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #64ffda !important;
        border-bottom-color: #64ffda !important;
    }
    
    /* FileUploader dropzone styling */
    .stFileUploader > div > div {
        background: rgba(5, 16, 36, 0.4);
        border: 1px dashed #8892b0;
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Render UI Header
st.title("🔍 Face Recognition & PCA Analysis Dashboard")
st.markdown("Sistem komparasi wajah dengan reduksi dimensi **PCA**, **Euclidean Distance**, dan **Cosine Similarity**.")

# Caching model PCA untuk mencegah re-fitting pada setiap interaksi UI
@st.cache_resource
def train_pca():
    with st.spinner("Fetching dataset and fitting PCA model..."):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        # Inisialisasi PCA dengan 50 principal components dan proses whitening
        pca = PCA(n_components=50, whiten=True)
        pca.fit(X)
        return pca, X

pca_model, dataset_matrix = train_pca()
st.success("✅ Model terinisialisasi.")

# Instansiasi object deteksi wajah berbasis Viola-Jones (Haar Cascade)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    # Parsing byte stream image ke array NumPy (tipe data uint8)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    # Dekode memori ke format matriks citra grayscale
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Deteksi bounding box wajah
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    # Ekstraksi Region of Interest (ROI) pada koordinat wajah
    face_crop = img[y:y+h, x:x+w]
    # Standarisasi resolusi input ke 64x64 piksel
    face_resize = cv2.resize(face_crop, (64, 64))
    
    # Transformasi matriks 2D (64, 64) menjadi vektor 1D (4096,)
    face_flat = face_resize.flatten()
    # Proyeksi vektor input ke PCA space (reduksi dari dimensi 4096 ke 50)
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_resize

# Inisialisasi komponen FileUploader menggunakan layout grid columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Image Source 1")
    file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"], key="f1")

with col2:
    st.subheader("📷 Image Source 2")
    file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"], key="f2")

# Trigger komputasi matriks saat button onClick event dipanggil
if st.button("⚖️ Run Analysis", type="primary", use_container_width=True):
    if file1 and file2:
        with st.spinner("Ekstraksi fitur PCA berjalan..."):
            vec1, img_crop1 = process_face(file1)
            vec2, img_crop2 = process_face(file2)
            
            if vec1 is None or vec2 is None:
                st.error("Gagal mendeteksi koordinat wajah pada citra input.")
            else:
                # Kalkulasi L2 Norm (Euclidean Distance) antar dua vektor PCA
                diff = vec1 - vec2
                euclidean_dist = np.sqrt(np.sum(diff ** 2))
                
                # Kalkulasi Cosine Similarity
                dot_product = np.dot(vec1, vec2)
                norm_v1 = np.linalg.norm(vec1)
                norm_v2 = np.linalg.norm(vec2)
                cosine_sim = dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) != 0 else 0.0
                
                # Konversi nilai cosine (range -1 hingga 1) ke persentase absolut
                similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
                
                # Render thresholding evaluasi kesamaan fitur
                st.markdown("---")
                if similarity_percentage >= 70:
                    st.success(f"### KESIMPULAN: COCOK (Similarity: {similarity_percentage}%)")
                else:
                    st.error(f"### KESIMPULAN: BERBEDA (Similarity: {similarity_percentage}%)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Kemiripan", f"{similarity_percentage}%")
                m2.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                m3.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
                
                # Render dashboard visualisasi data komparatif
                st.markdown("---")
                st.subheader("📊 Analisis Visual PCA (Principal Component Analysis)")
                
                tab1, tab2, tab3 = st.tabs(["Deteksi & Rekonstruksi", "Distribusi Vektor", "Statistik PCA"])
                
                with tab1:
                    st.write("**1. Output ROI Cropping (64x64 matrix)**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(img_crop1, caption="Input 1 (Asli)")
                    c2.image(img_crop2, caption="Input 2 (Asli)")
                    
                    # Transformasi inverse dari subspace PCA (50) kembali ke original space matrix (64x64)
                    rekon1 = pca_model.inverse_transform([vec1]).reshape(64, 64)
                    rekon2 = pca_model.inverse_transform([vec2]).reshape(64, 64)
                    
                    # Clipping dan casting hasil inverse transform ke tipe data uint8 untuk dirender
                    c3.image(np.clip(rekon1, 0, 255).astype(np.uint8), caption="Inverse Transform 1")
                    c4.image(np.clip(rekon2, 0, 255).astype(np.uint8), caption="Inverse Transform 2")

                with tab2:
                    st.write("**2. Mapping 50 Principal Components**")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Konfigurasi color scheme pada canvas figure matplotlib
                    fig.patch.set_facecolor('#051024')
                    ax.set_facecolor('#051024')
                    
                    x_axis = np.arange(50)
                    # Plot bar chart distribusi bobot vektor dari komponen PCA
                    ax.bar(x_axis - 0.2, vec1, 0.4, label='Vektor 1', color='#64ffda')
                    ax.bar(x_axis + 0.2, vec2, 0.4, label='Vektor 2', color='#ccd6f6')
                    
                    ax.set_xlabel("Principal Component Index", color='#8892b0')
                    ax.set_ylabel("Nilai Bobot", color='#8892b0')
                    ax.tick_params(colors='#8892b0')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#8892b0')
                    ax.legend(facecolor='#10223f', edgecolor='#8892b0', labelcolor='#ccd6f6')
                    
                    st.pyplot(fig)

                with tab3:
                    st.write("**3. Evaluasi Model PCA (Dataset-Level Metrics)**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Ekstraksi dan kalkulasi cumulative variance dari komponen
                        explained_variance = pca_model.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        fig2.patch.set_facecolor('#051024')
                        ax2.set_facecolor('#051024')
                        
                        ax2.plot(cumulative_variance, marker='o', linestyle='-', color='#64ffda')
                        ax2.set_title("Cumulative Explained Variance", color='#ccd6f6')
                        ax2.set_xlabel("N Komponen", color='#8892b0')
                        ax2.set_ylabel("Rasio Varians", color='#8892b0')
                        ax2.tick_params(colors='#8892b0')
                        for spine in ax2.spines.values():
                            spine.set_edgecolor('#8892b0')
                        ax2.grid(True, color='#10223f')
                        
                        st.pyplot(fig2)
                        st.caption(f"Retensi informasi citra: **{cumulative_variance[-1]*100:.1f}%** untuk n_components=50.")
                        
                    with col_b:
                        st.write("**Eigenface 0 (Principal Component Index 0)**")
                        # Reshape komponen eigenvector urutan pertama ke resolusi asal
                        eigenface_0 = pca_model.components_[0].reshape(64, 64)
                        
                        fig3, ax3 = plt.subplots(figsize=(3, 3))
                        fig3.patch.set_facecolor('#051024')
                        ax3.imshow(eigenface_0, cmap='gray')
                        ax3.axis('off')
                        
                        st.pyplot(fig3)
    else:
        st.warning("Exception: Harap unggah kedua file citra terlebih dahulu.")
