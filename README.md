# Machine-Learning-Olah-Data-Netflix
Kode jupyter notebook saya yang digunakan untuk olah data Netlix

# Link Dataset Machine Learning pada Dataset Netflix:
https://www.kaggle.com/datasets/paramvir705/netflix-dataset

## Deskripsi Proyek

Proyek ini menggunakan dataset Netflix Titles untuk melakukan analisis data dan membangun model Machine Learning yang dapat mengklasifikasikan jenis konten Netflix menjadi Movie atau TV Show.

Proses yang dilakukan meliputi pemuatan dataset, pembersihan data, pembentukan fitur, eksplorasi data, pembagian data latih dan data uji, pelatihan beberapa algoritma klasifikasi, evaluasi model, analisis fitur penting, penyimpanan model, serta peninjauan data berbentuk JSON.

## 1. Loading Data

Pada tahap awal, program menggunakan library Pandas untuk membaca dataset Netflix dari file `netflix_titles.csv`.

Perintah `df.head()` digunakan untuk menampilkan lima baris pertama dataset, sedangkan `df.info()` digunakan untuk mengetahui struktur data, jumlah baris, jumlah kolom, dan tipe data.

Dataset yang digunakan memiliki 8.807 baris dan 12 kolom, yaitu show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, dan description.

## 2. Data Cleaning

Tahap data cleaning dilakukan untuk memeriksa dan memperbaiki data sebelum digunakan dalam pemodelan.

Program memeriksa informasi dataset, statistik deskriptif, nilai kosong, dan data duplikat. Beberapa kolom seperti director, cast, country, date_added, rating, dan duration memiliki nilai kosong.

Kode juga memperbaiki tiga baris yang nilai durasinya masuk ke kolom rating. Nilai tersebut dipindahkan ke kolom duration agar struktur data menjadi lebih sesuai.

Nilai kosong pada kolom director, cast, country, dan rating diisi menggunakan label Unknown. Kolom date_added dikonversi menjadi format tanggal, kemudian nilai tanggal yang kosong diisi menggunakan median tanggal yang tersedia.

Pemeriksaan duplikat dilakukan menggunakan `duplicated()` dan penghapusan menggunakan `drop_duplicates()`. Berdasarkan output notebook, tidak ditemukan data duplikat sehingga jumlah data tetap 8.807 baris.

## 3. Feature Engineering

Feature engineering dilakukan untuk membentuk fitur yang dapat digunakan oleh algoritma Machine Learning.

Fitur yang dibuat meliputi:

- `type`: jenis konten yang menjadi target klasifikasi.
- `release_year`: tahun rilis konten.
- `year_added`: tahun konten ditambahkan ke Netflix.
- `month_added`: bulan konten ditambahkan.
- `rating`: kategori rating konten.
- `country_main`: negara utama yang tercatat.
- `n_countries`: jumlah negara yang tercatat.
- `n_cast`: jumlah nama pemeran.
- `n_genres`: jumlah kategori atau genre.
- `has_director`: penanda ketersediaan informasi sutradara.
- `desc_length`: panjang deskripsi.
- `title_length`: panjang judul.

Negara utama diambil dari data country. Sepuluh negara yang paling sering muncul dipertahankan, sedangkan negara lainnya dikelompokkan ke kategori Other.

Tujuan tahap ini adalah mengubah informasi awal menjadi fitur yang lebih terstruktur sehingga dapat digunakan dalam analisis dan pemodelan.

## 4. Analisis dan Penanganan Outlier

Program menggunakan box plot untuk melihat sebaran data dan mengidentifikasi kemungkinan outlier pada fitur release_year, year_added, n_cast, n_genres, dan desc_length.

Pemeriksaan jumlah outlier menggunakan metode Interquartile Range (IQR), yaitu metode yang memanfaatkan kuartil pertama dan kuartil ketiga untuk menentukan batas nilai yang berpotensi ekstrem.

Hasil pemeriksaan pada notebook menunjukkan jumlah outlier sebagai berikut:

- release_year: 719
- year_added: 56
- n_cast: 123
- n_genres: 0
- desc_length: 564

Pembatasan nilai atau capping diterapkan pada fitur n_cast dan desc_length. Sementara itu, nilai tahun rilis dan tahun penambahan tidak dibatasi karena nilai yang jauh dari mayoritas belum tentu merupakan kesalahan data.

## 5. Encoding Fitur

Tahap encoding mengubah data kategorikal menjadi bentuk numerik agar dapat diproses oleh model Machine Learning.

Target klasifikasi dikodekan menjadi:

- Movie = 0
- TV Show = 1

Kolom rating dan country_main diubah menggunakan metode One-Hot Encoding dengan fungsi `pd.get_dummies()`.

Hasilnya, variabel X berisi fitur masukan untuk model, sedangkan variabel y berisi target klasifikasi. Berdasarkan output notebook, X memiliki ukuran 8.807 baris dan 33 kolom fitur.

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis dilakukan untuk memahami karakteristik dan pola yang terdapat dalam dataset.

Notebook menghasilkan beberapa visualisasi, yaitu:

**Distribusi Movie dan TV Show**

Diagram batang digunakan untuk membandingkan jumlah film dan serial TV dalam dataset Netflix.

**Distribusi fitur numerik**

Histogram digunakan untuk melihat distribusi tahun rilis, tahun penambahan, bulan penambahan, jumlah negara, jumlah pemeran, jumlah genre, panjang deskripsi, dan panjang judul.

**Distribusi rating dan negara**

Count plot digunakan untuk membandingkan kategori rating dan negara utama berdasarkan jenis konten.

**Correlation Matrix**

Heatmap korelasi digunakan untuk melihat hubungan statistik antarfitur numerik dan target klasifikasi. Notebook juga menampilkan 15 fitur dengan korelasi absolut tertinggi terhadap target.

Visualisasi ini membantu memahami pola dalam dataset sebelum model dilatih. Namun, korelasi tidak secara langsung membuktikan hubungan sebab-akibat.

## 7. Splitting Data

Dataset dibagi menjadi data latih dan data uji menggunakan fungsi `train_test_split()`.

Pembagian dilakukan dengan komposisi 80% data latih dan 20% data uji. Parameter `random_state=1` digunakan agar pembagian data dapat direproduksi, sedangkan `stratify=y` membantu mempertahankan proporsi kelas.

Hasil pembagian data:

- Total data: 8.807
- Data latih: 7.045
- Data uji: 1.762

Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk mengevaluasi kemampuan model dalam memprediksi data yang tidak digunakan selama pelatihan.

## 8. Standardisasi Data

Standardisasi dilakukan menggunakan `StandardScaler` untuk fitur yang digunakan oleh Logistic Regression.

Scaler dilatih menggunakan data latih, kemudian digunakan untuk mentransformasi data latih dan data uji.

Standardisasi bertujuan menyamakan skala fitur agar perbedaan rentang nilai tidak terlalu memengaruhi proses pembelajaran model.

Pada notebook ini, standardisasi digunakan untuk Logistic Regression. Decision Tree, Random Forest, dan Gradient Boosting menggunakan fitur tanpa standardisasi.

## 9. Modeling

Notebook melatih empat algoritma klasifikasi untuk membedakan Movie dan TV Show.

**Logistic Regression**

Logistic Regression digunakan untuk memprediksi probabilitas suatu data termasuk ke dalam kelas Movie atau TV Show. Model ini menggunakan fitur yang telah distandardisasi.

**Decision Tree**

Decision Tree melakukan klasifikasi berdasarkan aturan keputusan yang dibentuk dari fitur. Model pada notebook menggunakan max_depth=6 untuk membatasi kedalaman pohon.

**Random Forest**

Random Forest menggabungkan prediksi dari banyak pohon keputusan. Notebook menggunakan 200 pohon melalui parameter `n_estimators=200`.

**Gradient Boosting Classifier**

Gradient Boosting membangun model secara bertahap. Setiap tahap berusaha memperbaiki kesalahan prediksi dari tahap sebelumnya.

Keempat algoritma tersebut kemudian dibandingkan berdasarkan hasil evaluasi pada data uji.

## 10. Hasil Evaluasi Model

Evaluasi dilakukan menggunakan Accuracy, Precision, Recall, dan F1-score.

Hasil yang ditampilkan pada notebook adalah sebagai berikut:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,9540 | 0,9332 | 0,9140 | 0,9235 |
| Decision Tree | 0,9552 | 0,9436 | 0,9065 | 0,9247 |
| Random Forest | 0,9597 | 0,9462 | 0,9196 | 0,9327 |
| Gradient Boosting | 0,9586 | 0,9477 | 0,9140 | 0,9305 |

Penjelasan metrik evaluasi:

- **Accuracy** mengukur proporsi prediksi yang benar dari seluruh data uji.
- **Precision** mengukur ketepatan prediksi pada kelas positif, yaitu TV Show.
- **Recall** mengukur proporsi data TV Show yang berhasil dikenali oleh model.
- **F1-score** menggabungkan precision dan recall menjadi satu nilai evaluasi.

Berdasarkan hasil tersebut, Random Forest memperoleh F1-score tertinggi sebesar 0,9327 dengan accuracy sebesar 0,9597 atau sekitar 95,97%.

Oleh karena itu, Random Forest dipilih sebagai model terbaik berdasarkan F1-score.

## 11. Classification Report dan Confusion Matrix

Setelah model terbaik dipilih, notebook menampilkan classification report dan confusion matrix.

Classification report memberikan informasi mengenai precision, recall, F1-score, dan jumlah data pada setiap kelas.

Berdasarkan output notebook, hasil klasifikasinya adalah:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Movie | 0,97 | 0,98 | 0,97 | 1.227 |
| TV Show | 0,95 | 0,92 | 0,93 | 535 |

Accuracy keseluruhan mencapai sekitar 96%.

Confusion matrix digunakan untuk membandingkan kelas sebenarnya dengan kelas hasil prediksi. Dengan visualisasi ini, jumlah prediksi benar maupun kesalahan klasifikasi untuk Movie dan TV Show dapat diamati.

## 12. Feature Importance

Notebook menampilkan sepuluh fitur terpenting menggunakan atribut `feature_importances_` dari model Gradient Boosting.

Grafik tersebut menunjukkan fitur yang berkontribusi terhadap proses pengambilan keputusan model Gradient Boosting.

Perlu diperhatikan bahwa feature importance yang ditampilkan berasal dari Gradient Boosting, sedangkan model terbaik berdasarkan F1-score adalah Random Forest. Dengan demikian, grafik tersebut menjelaskan kontribusi fitur pada Gradient Boosting, bukan secara langsung pada Random Forest.

## 13. Penyimpanan Model

Model terbaik disimpan ke dalam dua format file, yaitu:

- `best_model.joblib`
- `best_model.pkl`

File pertama dibuat menggunakan `joblib.dump()`, sedangkan file kedua dibuat menggunakan `pickle.dump()`.

Penyimpanan ini bertujuan agar model yang sudah dilatih dapat disimpan dan dimuat kembali tanpa harus mengulang seluruh proses pelatihan.

Pada notebook, model yang disimpan adalah model terbaik berdasarkan hasil evaluasi, yaitu Random Forest.

## 14. Data Berformat JSON

Pada cell terakhir notebook, terdapat sebuah objek data berbentuk JSON dengan key `data`.

Objek tersebut berisi satu baris data yang terdiri dari 76 nilai numerik. Nilai-nilai tersebut ditulis dalam bentuk array bersarang, yaitu sebuah list yang berisi list lainnya.

Secara umum, strukturnya berbentuk seperti berikut:

```json
{
  "data": [
    [0.0258, -0.9176, 0.7985, 0.0046]
  ]
}
