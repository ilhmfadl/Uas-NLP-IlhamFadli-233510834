"""
Modul Loader Dokumen PDF
========================

Modul ini bertanggung jawab untuk memuat seluruh dokumen PDF
dari direktori data/pdf/ menggunakan PyPDFLoader dari LangChain.

Proses:
1. Membaca setiap file PDF yang terdaftar di settings.PDF_FILES
2. Membersihkan teks (menghapus whitespace berlebih)
3. Menambahkan metadata sumber ke setiap dokumen
4. Mengembalikan list Document untuk proses selanjutnya
"""

import re
from langchain_community.document_loaders import PyPDFLoader
from src.config.settings import PDF_DIR, PDF_FILES


def clean_text(text: str) -> str:
    """
    Membersihkan teks dari karakter yang tidak diperlukan.

    Proses pembersihan meliputi:
    - Menghapus spasi berlebih (multiple spaces)
    - Menghapus baris kosong berlebih
    - Menghapus karakter kontrol yang tidak terlihat

    Args:
        text: Teks mentah dari PDF

    Returns:
        Teks yang sudah dibersihkan
    """
    # Hapus karakter kontrol (selain newline dan tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Ganti multiple spaces dengan single space
    text = re.sub(r' +', ' ', text)
    # Ganti lebih dari 2 baris kosong berturut-turut menjadi 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Hapus spasi di awal dan akhir setiap baris
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def load_pdf_documents() -> list:
    """
    Memuat seluruh dokumen PDF dari direktori yang dikonfigurasi.

    Fungsi ini akan:
    1. Mengiterasi setiap file PDF di PDF_FILES
    2. Memuat konten menggunakan PyPDFLoader
    3. Membersihkan teks setiap halaman
    4. Menambahkan metadata (nama file sumber)
    5. Menampilkan progress loading

    Returns:
        list: Daftar objek Document dari semua PDF

    Raises:
        FileNotFoundError: Jika file PDF tidak ditemukan
        ValueError: Jika tidak ada dokumen yang berhasil dimuat
    """
    all_documents = []

    print("\n📄 Memuat dokumen PDF...")
    print("-" * 50)

    for pdf_file in PDF_FILES:
        pdf_path = PDF_DIR / pdf_file

        # Validasi file ada
        if not pdf_path.exists():
            print(f"  ❌ File tidak ditemukan: {pdf_file}")
            raise FileNotFoundError(
                f"File PDF tidak ditemukan: {pdf_path}\n"
                f"Pastikan file '{pdf_file}' ada di direktori: {PDF_DIR}"
            )

        try:
            # Muat PDF menggunakan PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            # Validasi PDF tidak kosong
            if not documents:
                print(f"  ⚠️  PDF kosong: {pdf_file}")
                continue

            # Bersihkan teks dan tambahkan metadata
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata['source_file'] = pdf_file

                # Hapus dokumen dengan konten terlalu pendek (< 50 karakter)
                # karena biasanya hanya berisi header/footer
                if len(doc.page_content) < 50:
                    continue

                all_documents.append(doc)

            print(f"  ✅ {pdf_file}: {len(documents)} halaman dimuat")

        except Exception as e:
            print(f"  ❌ Gagal memuat {pdf_file}: {str(e)}")
            raise

    print("-" * 50)
    print(f"  📊 Total dokumen dimuat: {len(all_documents)} halaman")

    # Validasi ada dokumen yang berhasil dimuat
    if not all_documents:
        raise ValueError(
            "Tidak ada dokumen yang berhasil dimuat. "
            "Periksa file PDF di direktori data/pdf/"
        )

    return all_documents
