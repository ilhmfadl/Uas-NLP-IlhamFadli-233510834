"""
Modul Loader Data CSV Nutrisi
=============================

Modul ini memuat dan memproses data nutrisi dari file Nutrients.csv.

Dataset berisi informasi nutrisi makanan dengan kolom:
- Food: Nama makanan
- Measure: Ukuran porsi
- Grams: Berat dalam gram
- Calories: Kalori (kkal)
- Protein: Protein (gram)
- Fat: Lemak total (gram)
- Sat.Fat: Lemak jenuh (gram)
- Fiber: Serat (gram)
- Carbs: Karbohidrat (gram)
- Category: Kategori makanan

Catatan penting:
- Nilai 't' dalam dataset berarti 'trace' (jumlah sangat kecil, dibulatkan ke 0)
- Beberapa nilai numerik mengandung koma sebagai pemisah ribuan (contoh: '1,419')
"""

import pandas as pd
from src.config.settings import CSV_DIR, CSV_FILE

# Cache untuk DataFrame agar tidak perlu load berulang kali
_nutrition_data = None


def load_nutrition_data() -> pd.DataFrame:
    """
    Memuat data nutrisi dari file CSV dan melakukan preprocessing.

    Preprocessing meliputi:
    1. Membaca file CSV
    2. Menghapus whitespace berlebih dari nama kolom dan nilai
    3. Mengganti nilai 't' (trace) dengan 0
    4. Menghapus koma pemisah ribuan dari nilai numerik
    5. Mengkonversi kolom numerik ke tipe float

    Returns:
        pd.DataFrame: DataFrame yang sudah bersih dan siap digunakan

    Raises:
        FileNotFoundError: Jika file CSV tidak ditemukan
        ValueError: Jika file CSV kosong
    """
    global _nutrition_data

    # Gunakan cache jika sudah pernah dimuat
    if _nutrition_data is not None:
        return _nutrition_data

    csv_path = CSV_DIR / CSV_FILE

    # Validasi file ada
    if not csv_path.exists():
        raise FileNotFoundError(
            f"File CSV tidak ditemukan: {csv_path}\n"
            f"Pastikan file '{CSV_FILE}' ada di direktori: {CSV_DIR}"
        )

    # Muat CSV
    df = pd.read_csv(csv_path)

    # Validasi tidak kosong
    if df.empty:
        raise ValueError(f"File CSV kosong: {csv_path}")

    # Bersihkan nama kolom (hapus whitespace)
    df.columns = df.columns.str.strip()

    # Bersihkan nilai string (hapus whitespace)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # Kolom yang seharusnya numerik
    numeric_columns = ['Grams', 'Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']

    for col in numeric_columns:
        if col in df.columns:
            # Ganti 't' (trace) dengan 0
            df[col] = df[col].replace('t', '0')
            df[col] = df[col].replace('', '0')

            # Hapus koma pemisah ribuan (contoh: '1,419' -> '1419')
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)

            # Konversi ke float
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Cache hasil
    _nutrition_data = df

    return df


def search_food(query: str) -> pd.DataFrame:
    """
    Mencari makanan berdasarkan nama (case-insensitive, partial match).

    Args:
        query: Nama makanan yang dicari (sebagian atau lengkap)

    Returns:
        pd.DataFrame: Baris-baris yang cocok dengan pencarian.
                      DataFrame kosong jika tidak ditemukan.
    """
    df = load_nutrition_data()

    # Pencarian case-insensitive dan partial match
    mask = df['Food'].str.lower().str.contains(query.lower(), na=False)
    results = df[mask]

    return results


def get_all_food_names() -> list:
    """
    Mengembalikan daftar semua nama makanan yang tersedia.

    Returns:
        list: Daftar nama makanan unik
    """
    df = load_nutrition_data()
    return df['Food'].unique().tolist()


def get_categories() -> list:
    """
    Mengembalikan daftar semua kategori makanan.

    Returns:
        list: Daftar kategori unik
    """
    df = load_nutrition_data()
    return df['Category'].unique().tolist()
