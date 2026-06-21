"""
Modul Embedding dan Upload ke Pinecone
=======================================

Modul ini bertanggung jawab untuk:
1. Menginisialisasi model embedding (HuggingFace)
2. Mengelola koneksi ke Pinecone vector database
3. Mengupload chunks yang sudah di-embedding ke Pinecone

Model embedding yang digunakan:
- paraphrase-multilingual-MiniLM-L12-v2
- Mendukung 50+ bahasa termasuk Bahasa Indonesia
- Menghasilkan vektor 384 dimensi
- Optimal untuk pencarian semantik multilingual
"""

import os
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from src.config.settings import (
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL_NAME,
)

class SentenceTransformersCustomEmbeddings(Embeddings):
    """Custom Embeddings class using SentenceTransformer directly."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Generate embeddings and convert to list of floats
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


def get_embedding_model() -> Embeddings:
    """
    Menginisialisasi dan mengembalikan model embedding.

    Model: paraphrase-multilingual-MiniLM-L12-v2
    - Dimensi output: 384
    - Mendukung Bahasa Indonesia dan Inggris

    Returns:
        Embeddings: Instance model embedding yang siap digunakan
    """
    print("\n🧠 Memuat model embedding...")
    print(f"   Model: {EMBEDDING_MODEL_NAME}")

    embeddings = SentenceTransformersCustomEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("   ✅ Model embedding berhasil dimuat!")
    return embeddings



def initialize_pinecone() -> Pinecone:
    """
    Menginisialisasi koneksi ke Pinecone dan membuat index jika belum ada.

    Proses:
    1. Membuat client Pinecone menggunakan API key dari environment
    2. Memeriksa apakah index sudah ada
    3. Jika belum ada, membuat index baru dengan konfigurasi:
       - Nama: nutriguide-index
       - Dimensi: 384 (sesuai model embedding)
       - Metrik: cosine similarity
       - Spec: Serverless (AWS, us-east-1)
    4. Menunggu index siap digunakan

    Returns:
        Pinecone: Client Pinecone yang terkoneksi

    Raises:
        ValueError: Jika PINECONE_API_KEY tidak tersedia
        ConnectionError: Jika gagal terkoneksi ke Pinecone
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY belum diatur. "
            "Silakan atur di file .env"
        )

    print("\n🗄️  Menginisialisasi Pinecone...")

    try:
        # Buat client Pinecone
        pc = Pinecone(api_key=api_key)

        # Periksa apakah index sudah ada
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if PINECONE_INDEX_NAME in existing_indexes:
            print(f"   ✅ Index '{PINECONE_INDEX_NAME}' sudah ada")

            # Tampilkan statistik index
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print(f"   📊 Jumlah vektor saat ini: {stats.total_vector_count}")
        else:
            print(f"   🔨 Membuat index baru: '{PINECONE_INDEX_NAME}'...")

            # Buat index baru dengan ServerlessSpec
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,  # 384 dimensi
                metric=PINECONE_METRIC,         # cosine similarity
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,       # AWS
                    region=PINECONE_REGION,      # us-east-1
                ),
            )

            # Tunggu index siap (polling setiap 2 detik)
            print("   ⏳ Menunggu index siap...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
                time.sleep(2)

            print(f"   ✅ Index '{PINECONE_INDEX_NAME}' berhasil dibuat!")

        return pc

    except Exception as e:
        raise ConnectionError(
            f"Gagal terkoneksi ke Pinecone: {str(e)}\n"
            "Periksa API key dan koneksi internet Anda."
        )


def upload_to_pinecone(chunks: list, embedding_model: HuggingFaceEmbeddings) -> int:
    """
    Mengupload chunks yang sudah di-embedding ke Pinecone.

    Proses:
    1. Memecah chunks menjadi batch (100 per batch)
    2. Membuat embedding untuk setiap batch
    3. Mengupload ke Pinecone menggunakan PineconeVectorStore
    4. Menampilkan progress bar dengan tqdm

    Args:
        chunks: Daftar Document chunks yang akan diupload
        embedding_model: Model embedding yang digunakan

    Returns:
        int: Jumlah vektor yang berhasil diupload

    Raises:
        ConnectionError: Jika gagal upload ke Pinecone
    """
    print(f"\n📤 Mengupload {len(chunks)} chunks ke Pinecone...")
    print("-" * 50)

    try:
        # Inisialisasi client dan index Pinecone secara langsung
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)

        batch_size = 100
        
        # Proses upload dalam batch dengan progress bar
        for i in tqdm(range(0, len(chunks), batch_size),
                      desc="   Upload progress"):
            batch = chunks[i:i + batch_size]
            
            # Buat list teks untuk embedding batch
            texts = [doc.page_content for doc in batch]
            embeddings = embedding_model.embed_documents(texts)
            
            # Format vectors untuk upsert
            vectors_to_upsert = []
            for j, doc in enumerate(batch):
                # Buat ID unik untuk setiap chunk
                vector_id = f"chunk_{i + j}"
                metadata = {
                    "text": doc.page_content,
                    "source_file": doc.metadata.get("source_file", ""),
                    "page": doc.metadata.get("page", 0)
                }
                vectors_to_upsert.append((vector_id, embeddings[j], metadata))
                
            # Upsert ke Pinecone
            index.upsert(vectors=vectors_to_upsert, namespace=PINECONE_NAMESPACE)

        # Tunggu sebentar agar Pinecone meng-index semua vektor
        time.sleep(3)

        # Verifikasi jumlah vektor di Pinecone
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count

        print("-" * 50)
        print(f"   ✅ Upload selesai!")
        print(f"   📊 Total vektor di Pinecone: {vector_count}")

        return vector_count

    except Exception as e:
        raise ConnectionError(
            f"Gagal mengupload ke Pinecone: {str(e)}\n"
            "Periksa koneksi internet dan API key Anda."
        )
