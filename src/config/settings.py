"""
Modul Konfigurasi NutriGuide AI
================================

Modul ini memuat seluruh konfigurasi dan konstanta yang digunakan
di dalam sistem NutriGuide AI, termasuk:
- Path direktori proyek dan data
- Konfigurasi model LLM (Groq) dan Embedding (HuggingFace)
- Konfigurasi vector database (Pinecone)
- Parameter chunking dan retrieval
- Validasi environment variable yang diperlukan
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 1. Memuat Environment Variables dari file .env
# load_dotenv() akan mencari file .env di direktori root proyek
# dan memuat semua variabel ke dalam os.environ
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# 2. Konfigurasi Path / Direktori
# Semua path menggunakan pathlib agar kompatibel di Windows, Mac, dan Linux

# Direktori utama data
DATA_DIR = PROJECT_ROOT / "data"

# Direktori file PDF (sumber pengetahuan gizi)
PDF_DIR = DATA_DIR / "pdf"

# Direktori file CSV (data nutrisi makanan)
CSV_DIR = DATA_DIR / "csv"

# Daftar file PDF yang akan diproses sebagai knowledge base
PDF_FILES = [
    "PEDOMAN GIZI SEIMBANG.pdf",
    "healthy-diet-fact.pdf",
]

# Nama file CSV yang berisi data nutrisi makanan
CSV_FILE = "nutrients.csv"

# 3. Konfigurasi Pinecone (Vector Database)

# Pinecone digunakan untuk menyimpan dan mencari vektor embedding
# dari dokumen yang telah di-chunk

# Nama index di Pinecone
PINECONE_INDEX_NAME = "uas-nlp"

# Namespace untuk Pinecone
PINECONE_NAMESPACE = ""

# Metrik kesamaan vektor: cosine similarity cocok untuk teks
PINECONE_METRIC = "cosine"

# Dimensi vektor harus sesuai dengan model embedding yang digunakan
# paraphrase-multilingual-MiniLM-L12-v2 menghasilkan vektor 384 dimensi
PINECONE_DIMENSION = 384

# Cloud provider dan region untuk serverless Pinecone
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# 4. Konfigurasi Model Embedding (HuggingFace)

# Model embedding mengubah teks menjadi vektor numerik
# Model multilingual dipilih agar mendukung Bahasa Indonesia dan Inggris

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 5. Konfigurasi LLM (Groq + LLaMA)
# Groq menyediakan inferensi LLM yang sangat cepat
# LLaMA 3.3 70B dipilih karena kemampuan reasoning yang baik

# Nama model LLM yang digunakan di Groq
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# Temperature = 0 agar jawaban konsisten dan deterministik
# (tidak ada randomness dalam pemilihan token)
LLM_TEMPERATURE = 0

# 6. Konfigurasi Text Splitting / Chunking
# Dokumen dipecah menjadi potongan (chunk) kecil agar:
# - Embedding lebih fokus dan akurat
# - Retrieval mengembalikan bagian yang relevan saja

# Ukuran maksimal setiap chunk (dalam jumlah karakter)
CHUNK_SIZE = 800

# Jumlah karakter yang tumpang tindih antar chunk
# Overlap membantu menjaga konteks di batas antar chunk
CHUNK_OVERLAP = 200

# 7. Konfigurasi Retriever
# Retriever mengambil sejumlah chunk paling relevan dari vector DB

# Jumlah dokumen/chunk teratas yang diambil untuk setiap query
RETRIEVER_K = 5

# 8. Konfigurasi LangSmith (Tracing & Monitoring)
# LangSmith digunakan untuk memonitor dan men-debug pipeline LangChain

LANGCHAIN_PROJECT = "NutriGuideAI"

# 9. Daftar API Key yang Diperlukan
# Daftar environment variable yang WAJIB ada agar sistem berjalan

REQUIRED_API_KEYS = [
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "LANGCHAIN_API_KEY",
]


# 10. Fungsi Validasi Environment
def validate_environment() -> bool:
    """
    Memvalidasi bahwa semua environment variable yang diperlukan sudah tersedia.

    Fungsi ini memeriksa setiap API key di REQUIRED_API_KEYS dan mencetak
    status masing-masing (tersedia atau tidak). Jika ada yang belum diatur,
    fungsi akan menampilkan peringatan.

    Returns:
        bool: True jika semua API key tersedia, False jika ada yang hilang.
    """
    print("=" * 55)
    print(" NutriGuide AI - Validasi Environment")
    print("=" * 55)

    semua_tersedia = True

    for key in REQUIRED_API_KEYS:
        nilai = os.getenv(key)
        if nilai and nilai not in (
            "your_groq_api_key_here",
            "your_pinecone_api_key_here",
            "your_langchain_api_key_here",
        ):
            # Tampilkan hanya 8 karakter pertama untuk keamanan
            print(f"  ✅ {key}: {nilai[:8]}...")
        else:
            print(f"  ❌ {key}: BELUM DIATUR")
            semua_tersedia = False

    print("-" * 55)

    if semua_tersedia:
        print("  ✅ Semua API key sudah dikonfigurasi dengan benar!")
    else:
        print("  ⚠️  Beberapa API key belum diatur.")
        print("  📋 Salin file .env.example menjadi .env dan isi nilainya.")

    print("=" * 55)

    # Tampilkan ringkasan konfigurasi proyek
    print("\n📁 Konfigurasi Proyek:")
    print(f"   Root       : {PROJECT_ROOT}")
    print(f"   Data       : {DATA_DIR}")
    print(f"   PDF        : {PDF_DIR}")
    print(f"   CSV        : {CSV_DIR}")
    print(f"\n🤖 Model:")
    print(f"   LLM        : {LLM_MODEL_NAME}")
    print(f"   Embedding  : {EMBEDDING_MODEL_NAME}")
    print(f"   Temperature: {LLM_TEMPERATURE}")
    print(f"\n🗄️  Pinecone:")
    print(f"   Index      : {PINECONE_INDEX_NAME}")
    print(f"   Dimensi    : {PINECONE_DIMENSION}")
    print(f"   Metrik     : {PINECONE_METRIC}")
    print(f"   Cloud      : {PINECONE_CLOUD} ({PINECONE_REGION})")
    print(f"\n📄 Chunking:")
    print(f"   Chunk Size : {CHUNK_SIZE}")
    print(f"   Overlap    : {CHUNK_OVERLAP}")
    print(f"   Retriever K: {RETRIEVER_K}")

    return semua_tersedia


# ==============================================================
# Eksekusi validasi saat modul dijalankan langsung
# ==============================================================
if __name__ == "__main__":
    hasil = validate_environment()
    if not hasil:
        sys.exit(1)
