"""
NutriGuide AI - Pipeline Ingestion Utama
=========================================

Script ini mengorkestrasi seluruh pipeline ingestion data:
1. Validasi environment (API keys)
2. Memuat dokumen PDF dari direktori data/pdf/
3. Memecah dokumen menjadi chunks kecil
4. Membuat embedding dan mengupload ke Pinecone

Jalankan script ini dari root proyek:
    python ingest.py

Pastikan file .env sudah dikonfigurasi dengan benar sebelum menjalankan.
"""

import sys
import os
import time

# Fix untuk masalah crash PyTorch/SentenceTransformers pada Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
try:
    import gradio as gr
except ImportError:
    pass

# Konfigurasi encoding UTF-8 untuk output terminal Windows agar mendukung emoji
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import settings
from src.ingestion.loader import load_pdf_documents
from src.ingestion.chunker import create_chunks
from src.ingestion.embedder import (
    get_embedding_model,
    initialize_pinecone,
    upload_to_pinecone,
)


def main():
    """
    Fungsi utama yang menjalankan seluruh pipeline ingestion.

    Pipeline terdiri dari 5 tahap:
    1. Validasi environment variables (API keys)
    2. Memuat dokumen PDF
    3. Memecah dokumen menjadi chunks
    4. Menginisialisasi model embedding & Pinecone
    5. Mengupload chunks ke Pinecone

    Menampilkan waktu eksekusi dan ringkasan hasil di akhir proses.
    """
    # Catat waktu mulai untuk menghitung durasi total
    waktu_mulai = time.time()

    print("=" * 55)
    print("  🥗 NutriGuide AI - Pipeline Ingestion Data")
    print("=" * 55)
    print("  Memproses dokumen → chunk → embedding → Pinecone")
    print("=" * 55)

    # ===========================================================
    # TAHAP 1: Validasi Environment
    # ===========================================================
    print("\n📋 TAHAP 1/5: Validasi Environment")
    print("=" * 55)

    env_valid = settings.validate_environment()
    if not env_valid:
        print("\n❌ Environment belum lengkap. Proses dihentikan.")
        print("   Silakan lengkapi konfigurasi di file .env terlebih dahulu.")
        sys.exit(1)

    # ===========================================================
    # TAHAP 2: Memuat Dokumen PDF
    # ===========================================================
    print("\n📋 TAHAP 2/5: Memuat Dokumen PDF")
    print("=" * 55)

    try:
        dokumen = load_pdf_documents()
        total_dokumen = len(dokumen)
    except FileNotFoundError as e:
        print(f"\n❌ File tidak ditemukan: {e}")
        print("   Pastikan semua file PDF tersedia di direktori data/pdf/")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Kesalahan data: {e}")
        sys.exit(1)

    # ===========================================================
    # TAHAP 3: Chunking Dokumen
    # ===========================================================
    print("\n📋 TAHAP 3/5: Memecah Dokumen Menjadi Chunks")
    print("=" * 55)

    try:
        chunks = create_chunks(dokumen)
        total_chunks = len(chunks)
    except ValueError as e:
        print(f"\n❌ Kesalahan chunking: {e}")
        sys.exit(1)

    # ===========================================================
    # TAHAP 4: Inisialisasi Model Embedding & Pinecone
    # ===========================================================
    print("\n📋 TAHAP 4/5: Inisialisasi Embedding & Pinecone")
    print("=" * 55)

    try:
        # Muat model embedding dari HuggingFace
        model_embedding = get_embedding_model()

        # Inisialisasi koneksi ke Pinecone (buat index jika belum ada)
        initialize_pinecone()
    except ValueError as e:
        print(f"\n❌ Kesalahan konfigurasi: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n❌ Gagal terkoneksi: {e}")
        print("   Periksa koneksi internet dan API key Anda.")
        sys.exit(1)

    # ===========================================================
    # TAHAP 5: Upload ke Pinecone
    # ===========================================================
    print("\n📋 TAHAP 5/5: Upload Chunks ke Pinecone")
    print("=" * 55)

    try:
        total_vektor = upload_to_pinecone(chunks, model_embedding)
    except ConnectionError as e:
        print(f"\n❌ Gagal upload: {e}")
        sys.exit(1)

    # ===========================================================
    # Ringkasan Hasil
    # ===========================================================
    waktu_selesai = time.time()
    durasi = waktu_selesai - waktu_mulai

    # Konversi durasi ke menit dan detik
    menit = int(durasi // 60)
    detik = int(durasi % 60)

    print("\n" + "=" * 55)
    print("  ✅ PIPELINE INGESTION SELESAI!")
    print("=" * 55)
    print(f"  📄 Total dokumen (halaman) : {total_dokumen}")
    print(f"  ✂️  Total chunks            : {total_chunks}")
    print(f"  📊 Total vektor di Pinecone: {total_vektor}")
    print(f"  ⏱️  Waktu eksekusi          : {menit} menit {detik} detik")
    print("=" * 55)
    print("  🚀 Data siap digunakan untuk NutriGuide AI!")
    print("=" * 55)


# ==============================================================
# Entry Point
# ==============================================================
if __name__ == "__main__":
    main()
