# Eksperimen Sistem Preprocessing Otomatis untuk Prediksi Kondisi Pipa
Proyek Machine Learning ini berfokus pada otomatisasi proses preprocessing data untuk prediksi kondisi pipa, lengkap dengan integrasi GitHub Actions untuk otomatisasi pipeline. Project ini juga dirancang untuk environment otomatis lokal & cloud.

## Setup Environment
1. **Menggunakan Conda (Disarankan)**
   ```
   conda create -n MSML python=3.12.7 -y
   conda activate MSML
   ```
2. **Install Dependencies**
   ```
   pip install pandas numpy scikit-learn joblib
   ```

## Struktur Proyek
```
Eksperimen_SML_mrickyrizaldi/
├── dataset_raw/
│   └── market_pipe_thickness_loss_dataset.csv     # Dataset mentah
├── preprocessing/
│   ├── automate_mrickyrizaldi.py                  # Skrip preprocessing otomatis
│   ├── Eksperimen_MSML_mrickyrizaldi.ipynb        # Notebook eksperimen manual
│   └── preprocessed_data_auto_*/                  # Folder hasil preprocessing otomatis
├── .github/
│   └── workflows/
│       └── preprocessing.yml                      # GitHub Actions workflow
├── README.md                                      # Dokumentasi proyek ini
```

## Menjalankan Preprocessing Otomatis
1. **Basic Usage**
   ```
   python preprocessing/automate_mrickyrizaldi.py
   ```
2. **Output**
   ```
   preprocessing/preprocessed_data_auto_[timestamp]/
   ```

## Tahapan Preprocessing Otomatis
Script ```automate_mrickyrizaldi.py``` secara otomatis menjalankan tahapan berikut:
1. **Load Data:** Membaca dataset dari ```dataset_raw/```.
2. **Feature Cleaning:** Menghapus kolom ```Pipe_Size_mm``` jika ada.
3. **Feature Engineering:**
   - Identifikasi fitur numerik & kategorikal.
   - Log transformasi untuk fitur dengan distribusi skewed.
4. **Imputation & Scaling:** Imputasi & normalisasi numerik.
5. **Categorical Encoding:** One-hot encoding untuk fitur kategorikal.
6. **Label Encoding:** Encoding target (Condition).
7. **Train-Test Split:** Membagi data ke training & testing.
8. **Saving:** Menyimpan dataset hasil preprocessing, pipeline, dan encoder.

## GitHub Actions Workflow
Workflow otomatis di ```.github/workflows/preprocessing.yml``` akan:
1. Setup Python 3.12.7
2. Install dependencies
3. Menjalankan script preprocessing otomatis
4. Upload hasil sebagai artifact di GitHub Actions
5. Commit folder hasil ke repository (jika ada perubahan)

## Trigger:
1. Otomatis saat file dataset atau script preprocessing diubah
2. Manual via GitHub Actions tab
3. Saat Pull Request dibuat ke main
