"""
Modul Text Chunking
===================

Modul ini memecah dokumen menjadi potongan-potongan kecil (chunks)
menggunakan RecursiveCharacterTextSplitter dari LangChain.

Pemilihan Parameter Chunking:
-----------------------------
1. chunk_size = 800 karakter
   - Dokumen nutrisi umumnya memiliki paragraf pendek hingga menengah
   - Ukuran 800 karakter memberikan keseimbangan antara:
     * Mempertahankan konteks yang cukup dalam satu chunk
     * Menjaga spesifisitas agar retrieval lebih akurat
   - Terlalu kecil (< 500): konteks terpotong, jawaban tidak lengkap
   - Terlalu besar (> 1500): retrieval kurang fokus, noise meningkat

2. chunk_overlap = 200 karakter (~25% dari chunk_size)
   - Overlap memastikan informasi di batas antar chunk tidak hilang
   - 25% overlap adalah standar industri untuk dokumen teknis
   - Mencegah terpotongnya kalimat penting di perbatasan chunk
   - Membantu menjaga koherensi konteks saat retrieval

3. separators = ["\\n\\n", "\\n", ". ", " ", ""]
   - Prioritas pemisahan mengikuti hierarki struktur teks:
     * \\n\\n: batas paragraf (prioritas tertinggi)
     * \\n: batas baris
     * ". ": batas kalimat
     * " ": batas kata
     * "": karakter per karakter (fallback terakhir)
   - Ini memastikan chunk dipecah di batas alami teks
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(documents: list) -> list:
    """
    Memecah dokumen menjadi chunks menggunakan RecursiveCharacterTextSplitter.

    Args:
        documents: List objek Document dari proses loading

    Returns:
        list: Daftar chunks (Document objects) yang siap di-embedding

    Raises:
        ValueError: Jika input dokumen kosong
    """
    if not documents:
        raise ValueError("Daftar dokumen kosong. Tidak ada yang bisa di-chunk.")

    print("\n✂️  Memecah dokumen menjadi chunks...")
    print("-" * 50)

    # Inisialisasi text splitter dengan parameter yang telah dipilih
    # RecursiveCharacterTextSplitter memecah teks secara rekursif
    # menggunakan daftar separator dari yang paling diinginkan
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Maksimal 800 karakter per chunk
        chunk_overlap=CHUNK_OVERLAP,  # 200 karakter overlap antar chunk
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritas pemisahan
        length_function=len,          # Fungsi pengukur panjang teks
        is_separator_regex=False,     # Separator bukan regex
    )

    # Lakukan splitting pada seluruh dokumen
    chunks = text_splitter.split_documents(documents)

    # Hitung statistik chunks
    total_chunks = len(chunks)
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_length = total_chars / total_chunks if total_chunks > 0 else 0

    # Tampilkan statistik
    print(f"  📊 Konfigurasi Chunking:")
    print(f"     - Chunk Size   : {CHUNK_SIZE} karakter")
    print(f"     - Chunk Overlap: {CHUNK_OVERLAP} karakter")
    print(f"  📊 Hasil Chunking:")
    print(f"     - Total Chunks       : {total_chunks}")
    print(f"     - Rata-rata Panjang  : {avg_length:.0f} karakter")
    print(f"     - Total Karakter     : {total_chars:,}")
    print("-" * 50)

    return chunks
