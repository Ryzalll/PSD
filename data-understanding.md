# Data Understanding

## 1. Deskripsi Dataset

Dataset yang digunakan adalah "Healthcare Dataset" yang bersumber dari Kaggle. Dataset ini terdiri dari 10.000 baris dan 15 kolom.

**Kolom-kolom utama meliputi:**
- `Name`: Nama pasien
- `Age`: Usia pasien
- `Gender`: Jenis kelamin
- `Medical Condition`: Kondisi medis utama
- `Date of Admission`: Tanggal masuk rumah sakit
- `Billing Amount`: Jumlah tagihan (dalam format notasi ilmiah dengan pemisah koma)
- `Admission Type`: Jenis admisi (misalnya, 'Urgent', 'Emergency', 'Elective')
- `Medication`: Obat yang diberikan
- `Test Results`: Hasil tes medis

---

## 2. Load Dataset

Dataset yang digunakan akan diload ke PostGreSQL, tahapan seperti berikut : 

- **Pahami Tipe Data yang ada** Memahami tipe data pada setiap kolom untuk melakukan tahapan import dataset ke postgreSQL.
Setelah mengamati dataset yang ada, coba untuk membuat tabel baru di dataset yang ada di PostGreSQL.
dengan syntax seperti berikut : 
```{code}
    CREATE TABLE healthcare_data (
    Name VARCHAR(255),
    Age INTEGER,
    Gender VARCHAR(50),
    Blood_Type VARCHAR(10),
    Medical_Condition VARCHAR(255),
    Date_of_Admission DATE,
    Doctor VARCHAR(255),
    Hospital VARCHAR(255),
    Insurance_Provider VARCHAR(255),
    Billing_Amount VARCHAR(255), -- Diubah sementara menjadi teks
    Room_Number INTEGER,
    Admission_Type VARCHAR(50),
    Discharge_Date DATE,
    Medication VARCHAR(255),
    Test_Results VARCHAR(50)
);
```

- Lakukan proses impor melalui pgAdmin:

- Klik kanan pada tabel **healthcare_data** yang baru.

- Pilih Import/Export....

- Pastikan toggle pada Import.

- Pilih file **healthcare_dataset.csv**.

- Di tab Options, pastikan Header dicentang dan Delimiter adalah ; (semicolon).


- **Mengoptimalisasikan tipe data kolom:** 
    - Ganti semua koma (,) menjadi titik (.) di kolom Billing_Amount.
    - Ubah tipe data kolom Billing_Amount menjadi NUMERIC.
```{code}
UPDATE healthcare_data
SET Billing_Amount = REPLACE(Billing_Amount, ',', '.');
```
```{code}
ALTER TABLE healthcare_data
ALTER COLUMN Billing_Amount TYPE NUMERIC USING Billing_Amount::NUMERIC;
```

## 3. Connect ke Power BI

- Pergi ke menu **Home**, dan pilih **Get Data**

```{image} _images/getdata.png
:alt: getdata
:class: mb-1
:width: 800px
:align: center
```

- Kemudian pilih, **PostGreSQL** 
- Masukkan nama **Server** dan **Database** yang sudah dibuat sebelumnya
- Kemudian pilih nama tabel yang sudah dibuat **healthcare_data**
- Hasilnya akan seperti gambar di bawah
```{image} _images/datas.png
:alt: data
:class: mb-1
:width: 300px
:align: center
```