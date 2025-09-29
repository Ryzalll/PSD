# Data Understanding

## 1. Deskripsi Dataset

Data Understanding atau yang biasa disebut dengan Memahami Data adalah salah satu tahap penting dalam proses Knowledge Discovery in Databases (KDD) atau Data Sains. Ini merupakan langkah awal yang bertujuan untuk memahami data secara mendalam sebelum melakukan analisis lebih lanjut. Pemahaman data yang baik sangat krusial karena akan memengaruhi keberhasilan seluruh proses data mining.

**Kolom-kolom utama meliputi:**
- `sepal_length` → Decimal Number (cm)

- `sepal_width` → Decimal Number (cm)

- `petal_length` → Decimal Number (cm)

- `petal_width` → Decimal Number (cm)

- `species` → Text (kategori: setosa, versicolor, virginica)

---

## 2. Load Dataset

Pertama-tama pastikan data sudah siap ada di power BI

### 2.1 Tambahkan data

#### 2.1.1 Import data lewat PostGreSQL
- **Pahami Tipe Data yang ada** Memahami tipe data pada setiap kolom untuk melakukan tahapan import dataset ke postgreSQL.
Setelah mengamati dataset yang ada, coba untuk membuat tabel baru di dataset yang ada di PostGreSQL.
dengan syntax seperti berikut : 

```{code}
    CREATE TABLE iris_dataset (
    id SERIAL PRIMARY KEY,
    species VARCHAR(50),
    sepal_length NUMERIC,
    sepal_width NUMERIC,
    petal_length NUMERIC,
    petal_width NUMERIC
);
```

Kemudian tambahkan code SQL berikut : 

```{code}
    COPY public.iris_dataset(species, sepal_length, sepal_width, petal_length, petal_width) 
    FROM 'D:/Kuliah/Matkul/SMT 5/PSD/Dataset/iris-full.csv' 
    WITH (FORMAT csv, DELIMITER ',', HEADER, QUOTE '"', ESCAPE '"');
```


#### 2.1.2 Import data lewat MySQL

* Import dataset dengan format .csv ke dalam phpMyAdmin
* Pilih file format ".csv" kemudian centang Baris pertama adalah header
* Kemudian, Import

### 2.2 Load Dataset 

#### 2.2.1 Dengan Power BI

* Buka Power BI 
* Pilih Get Data
* Pilih python script

    Kemudian masukkan code berikut untuk mengambil dataset yang ada di PostgreSQL dan MySQL
```{note}
Kolom **id**, **species**, **sepal_length** berasal dari dataset MySQL, sedangkan untuk

Kolom **sepal_width**, **petal_length**, **petal_width** berasal dari dataset PostGreSQL
```

```{code}

import pandas as pd
import mysql.connector
import psycopg2

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="psd_dataset",
    port=3306
)

query = """
SELECT 
    id,
    Class AS species,
    `sepal length`  AS sepal_length,
    `sepal width`   AS sepal_width,
    `petal length`  AS petal_length,
    `petal width`   AS petal_width
FROM iris_full
"""

my_df_mysql = pd.read_sql(query, conn)
conn.close()

conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="1234",
    database="psd_dataset",
    port=5432
)

query = """
SELECT 
    id,
    species,
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
FROM public.iris_dataset
"""

pg_df = pd.read_sql(query, conn)


kolom_mysql = ["id","species","sepal_length"]

kolom_pgsql = [c for c in pg_df.columns if c not in kolom_mysql]

dataset = pd.concat([my_df_mysql[kolom_mysql], pg_df[kolom_pgsql]], axis = 1)
```

#### 2.2.1 Dengan VS Code

* Buat file dengan nama **connection.ipynb**
* Tuliskan code berikut untuk menambahkan dataset ke vscode

```{code}
import pandas as pd
import mysql.connector
import psycopg2

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="psd_dataset",
    port=3306
)

query = """
SELECT 
    id,
    Class AS species,
    `sepal length`  AS sepal_length,
    `sepal width`   AS sepal_width,
    `petal length`  AS petal_length,
    `petal width`   AS petal_width
FROM iris_full
"""

my_df_mysql = pd.read_sql(query, conn)
conn.close()

conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="1234",
    database="psd_dataset",
    port=5432
)

query = """
SELECT 
    id,
    species,
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
FROM public.iris_dataset
"""

pg_df = pd.read_sql(query, conn)


kolom_mysql = ["id","species","sepal_length"]

kolom_pgsql = [c for c in pg_df.columns if c not in kolom_mysql]

dataset = pd.concat([my_df_mysql[kolom_mysql], pg_df[kolom_pgsql]], axis = 1)

```

Data yang diperoleh ditampung dalam variabel dataframe dengan nama **dataset**
