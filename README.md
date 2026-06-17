Aplikasi web interaktif berbasis **Streamlit** yang mendemonstrasikan implementasi algoritma *Principal Component Analysis* (PCA) untuk reduksi dimensi dan pengenalan wajah (*Face Recognition*). Sistem ini mengekstrak fitur matematis esensial dari dua citra wajah dan mengkomparasinya menggunakan metrik jarak vektor.

---

## 🚀 Fitur Utama

* **Deteksi Wajah Otomatis:** Menggunakan algoritma *Viola-Jones* (OpenCV Haar Cascade) untuk mengisolasi Region of Interest (ROI) wajah dari citra asli.
* **Reduksi Dimensi Tingkat Lanjut:** Memampatkan citra beresolusi 64x64 piksel (4096 dimensi) menjadi hanya 50 *Principal Components* yang paling representatif tanpa kehilangan informasi krusial.
* **Analisis Kemiripan Akurat:** Mengevaluasi tingkat kecocokan dua wajah berdasarkan vektor fitur menggunakan komputasi *Euclidean Distance* (L2 Norm) dan *Cosine Similarity*.
* **Visualisasi Komprehensif:** Dilengkapi *dashboard* analitik untuk merender rekonstruksi *inverse transform*, distribusi bobot komponen, grafik *Cumulative Variance*, dan *Eigenfaces*.
* **Custom UI & DOM Manipulation:** Antarmuka dengan tema *Deep Ocean* terintegrasi efek *Glassmorphism* dan injeksi elemen HTML5 Canvas untuk animasi latar belakang interaktif.

## 🛠️ Teknologi & Pustaka (Tech Stack)

Sistem ini dikembangkan menggunakan ekosistem Python modern untuk pemrosesan data dan antarmuka web:
* **[Streamlit](https://streamlit.io/):** *Framework* utama untuk *rendering dashboard* interaktif.
* **[OpenCV](https://opencv.org/) (`cv2`):** Pemrosesan citra (konversi matriks, ekstraksi piksel, dan *Haar Cascades*).
* **[Scikit-Learn](https://scikit-learn.org/):** Pelatihan model PCA dan pengambilan dataset latih standar (*Olivetti Faces*).
* **[NumPy](https://numpy.org/):** Operasi aljabar linear dan manipulasi matriks multidimensi.
* **[Matplotlib](https://matplotlib.org/):** *Plotting* grafik matematis dan *rendering* citra *Eigenface*.

## ⚙️ Alur Kerja Sistem (How it Works)

1.  **Input:** Pengguna mengunggah dua citra (Masa Lalu & Sekarang) berekstensi `.jpg` / `.png`.
2.  **Pre-processing:** Citra dikonversi ke format *Grayscale*. Bounding box wajah dideteksi, di-*crop*, dan di-resize secara konstan ke matriks dimensi (64, 64).
3.  **Feature Extraction:** Matriks 2D di-*flatten* menjadi array 1D (4096 elemen). Array tersebut diproyeksikan ke dalam ruang PCA untuk menghasilkan vektor matriks berukuran 50.
4.  **Output & Evaluasi:** Komputasi jarak antara dua matriks vektor dilakukan. Persentase probabilitas kesamaan ditampilkan bersamaan dengan grafik visual dari kinerja model di *tab dashboard*.
