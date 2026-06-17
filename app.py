import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Inisialisasi parameter top-level antarmuka Streamlit
st.set_page_config(page_title="Face Recognition PCA", page_icon="🔍", layout="wide")

# Injeksi CSS terstruktur untuk modifikasi hierarki DOM dan implementasi UI Glassmorphism
custom_css = """
<style>
    /* Modifikasi background global ke skema warna gradient konstan */
    .stApp {
        background: linear-gradient(135deg, #10223f, #051024);
        color: #ccd6f6;
    }
    
    /* Terminasi visibilitas komponen default Streamlit (Hamburger, Footer, Top Padding) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* Standarisasi properti tipografi */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ccd6f6 !important;
    }
    
    /* Styling pseudo-classes pada komponen button (Hover states & Transitions) */
    div.stButton > button:first-child {
        background-color: transparent !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        border-radius: 5px;
        transition: all 0.3s ease;
        font-weight: 600;
        letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        background-color: rgba(100, 255, 218, 0.15) !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.3) !important;
        transform: translateY(-2px);
    }
    
    /* Modifikasi container metrics dengan properti backdrop-filter untuk efek Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(5, 16, 36, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #64ffda !important;
    }
    
    /* Modifikasi state UI pada arsitektur komponen Tabs */
    button[data-baseweb="tab"] {
        color: #8892b0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #64ffda !important;
        border-bottom-color: #64ffda !important;
    }
    
    /* Modifikasi area drag-and-drop file uploader */
    .stFileUploader > div > div {
        background: rgba(5, 16, 36, 0.4);
        border: 1px dashed #8892b0;
        border-radius: 10px;
    }
    
    /* Kelas CSS kustom untuk rendering custom alert element */
    .custom-alert-success {
        background: rgba(100, 255, 218, 0.1);
        border-left: 5px solid #64ffda;
        padding: 15px;
        border-radius: 5px;
        color: #64ffda;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    .custom-alert-error {
        background: rgba(255, 82, 82, 0.1);
        border-left: 5px solid #ff5252;
        padding: 15px;
        border-radius: 5px;
        color: #ff5252;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Render abstraksi header
st.title("🔍 Face Recognition & PCA Analysis Dashboard")
st.markdown("Sistem komparasi wajah dengan reduksi dimensi **PCA**, **Euclidean Distance**, dan **Cosine Similarity**.")

# Komputasi model PCA dan isolasi ke dalam memori cache
@st.cache_resource
def train_pca():
    with st.spinner("Fetching dataset and fitting PCA model..."):
        dataset = fetch_olivetti_faces()
        X = dataset.data
        pca = PCA(n_components=50, whiten=True)
        pca.fit(X)
        return pca, X

pca_model, dataset_matrix = train_pca()

# Instansiasi object deteksi Viola-Jones (Haar Cascade)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def process_face(image_file):
    # Parsing byte stream image ke array NumPy 
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Normalisasi iluminasi lokal (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Spatial smoothing (Gaussian Blur)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Ekstraksi Region of Interest (ROI) wajah
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
        
    (x, y, w, h) = faces[0]
    
    # Reduksi margin (Tighter Crop) untuk meminimalisasi noise pada arsitektur data latar
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    face_crop = img[y+margin_y : y+h-margin_y, x+margin_x : x+w-margin_x]
    
    # Downsampling resolusi matriks spasial
    face_resize = cv2.resize(face_crop, (64, 64))
    
    # Flattening array multidimensi dan min-max scaling [0, 1]
    face_flat = face_resize.flatten() / 255.0 
    
    # Proyeksi vektor input ke subspace PCA
    face_pca = pca_model.transform([face_flat])[0]
    
    return face_pca, face_resize

# Segmentasi layout UI via instance kolom dan container
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Image Source 1")
    file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"], key="f1")

with col2:
    st.subheader("📷 Image Source 2")
    file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"], key="f2")

# Parameter tuning threshold dinamis
st.markdown("---")
threshold = st.slider("⚙️ Threshold Toleransi Kesamaan (%)", min_value=60, max_value=95, value=75, step=1)

# Trigger komputasi matriks
if st.button("⚖️ Run Analysis", type="primary", use_container_width=True):
    if file1 and file2:
        with st.spinner("Mengeksekusi ekstraksi fitur PCA..."):
            vec1, img_crop1 = process_face(file1)
            vec2, img_crop2 = process_face(file2)
            
            if vec1 is None or vec2 is None:
                # Fallback rendering untuk exception handling
                st.markdown('<div class="custom-alert-error">Exception: Gagal mendeteksi koordinat wajah pada citra input. Evaluasi kontras spasial citra.</div>', unsafe_allow_html=True)
            else:
                # Komputasi Euclidean Distance (L2 Norm)
                diff = vec1 - vec2
                euclidean_dist = np.sqrt(np.sum(diff ** 2))
                
                # Komputasi Cosine Similarity (Dot Product)
                dot_product = np.dot(vec1, vec2)
                norm_v1 = np.linalg.norm(vec1)
                norm_v2 = np.linalg.norm(vec2)
                cosine_sim = dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) != 0 else 0.0
                
                # Konversi skala kosinus ke probabilitas absolut
                similarity_percentage = int(((cosine_sim + 1) / 2) * 100)
                
                # Rendering Custom HTML Output Alert berdasarkan komparasi logical
                st.markdown("---")
                if similarity_percentage >= threshold:
                    st.markdown(f'<div class="custom-alert-success">KESIMPULAN: WAJAH IDENTIK ({similarity_percentage}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="custom-alert-error">KESIMPULAN: WAJAH BERBEDA ({similarity_percentage}%)</div>', unsafe_allow_html=True)
                
                # Rendering blok metric spasial
                m1, m2, m3 = st.columns(3)
                m1.metric("Tingkat Kesamaan", f"{similarity_percentage}%")
                m2.metric("Cosine Similarity", f"{cosine_sim:.4f}")
                m3.metric("Euclidean Distance", f"{euclidean_dist:.4f}")
                
                # Isolasi scope render visual ke dalam komponen Tabulasi
                st.markdown("---")
                st.subheader("📊 Visualisasi Fitur PCA")
                
                tab1, tab2, tab3 = st.tabs(["Deteksi & Rekonstruksi", "Distribusi Vektor", "Statistik Matriks"])
                
                with tab1:
                    st.write("**1. Output Pre-processing ROI (CLAHE + Gaussian Blur)**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(img_crop1, caption="Input 1 (Processed)")
                    c2.image(img_crop2, caption="Input 2 (Processed)")
                    
                    # Inverse transformasi arsitektur PCA ke topologi ruang spasial
                    rekon1 = pca_model.inverse_transform([vec1]).reshape(64, 64)
                    rekon2 = pca_model.inverse_transform([vec2]).reshape(64, 64)
                    
                    # Casting dan clipping float matrix ke representasi grafis (uint8)
                    c3.image(np.clip(rekon1 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform 1")
                    c4.image(np.clip(rekon2 * 255, 0, 255).astype(np.uint8), caption="Inverse Transform 2")

                with tab2:
                    st.write("**2. Proyeksi Spasial 50 Principal Components**")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    # Override color profile canvas Matplotlib
                    fig.patch.set_facecolor('#051024')
                    ax.set_facecolor('#051024')
                    
                    x_axis = np.arange(50)
                    ax.bar(x_axis - 0.2, vec1, 0.4, label='Vektor 1', color='#64ffda')
                    ax.bar(x_axis + 0.2, vec2, 0.4, label='Vektor 2', color='#ccd6f6')
                    
                    ax.set_xlabel("Indeks Komponen", color='#8892b0')
                    ax.set_ylabel("Magnitudo", color='#8892b0')
                    ax.tick_params(colors='#8892b0')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#8892b0')
                    ax.legend(facecolor='#10223f', edgecolor='#8892b0', labelcolor='#ccd6f6')
                    st.pyplot(fig)

                with tab3:
                    st.write("**3. Metrik Evaluasi Model PCA (Cumulative Ratio)**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Ekstraksi dan komputasi kumulatif varians dari set data pelatihan
                        explained_variance = pca_model.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        fig2.patch.set_facecolor('#051024')
                        ax2.set_facecolor('#051024')
                        
                        ax2.plot(cumulative_variance, marker='o', linestyle='-', color='#64ffda')
                        ax2.set_title("Cumulative Explained Variance", color='#ccd6f6')
                        ax2.set_xlabel("n_components", color='#8892b0')
                        ax2.set_ylabel("Rasio Varians", color='#8892b0')
                        ax2.tick_params(colors='#8892b0')
                        for spine in ax2.spines.values():
                            spine.set_edgecolor('#8892b0')
                        ax2.grid(True, color='#10223f')
                        st.pyplot(fig2)
                        st.caption(f"Retensi informasi spasial: **{cumulative_variance[-1]*100:.1f}%**.")
                        
                    with col_b:
                        st.write("**Topologi Eigenface (Index 0)**")
                        # Rendering komponen struktural paling dominan
                        eigenface_0 = pca_model.components_[0].reshape(64, 64)
                        fig3, ax3 = plt.subplots(figsize=(3, 3))
                        fig3.patch.set_facecolor('#051024')
                        ax3.imshow(eigenface_0, cmap='gray')
                        ax3.axis('off')
                        st.pyplot(fig3)
    else:
        # Fallback requirement block
        st.markdown('<div class="custom-alert-error">Harap integrasikan kedua input file citra sebelum inisialisasi proses komputasi.</div>', unsafe_allow_html=True)
