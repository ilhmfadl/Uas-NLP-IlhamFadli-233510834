"""
Modul Retriever untuk Pinecone Vector Database
===============================================

Modul ini menghubungkan NutriGuide AI dengan Pinecone
untuk melakukan pencarian semantik pada dokumen yang telah di-embedding.

Proses retrieval:
1. Menerima query dari pengguna
2. Mengubah query menjadi vektor menggunakan model embedding
3. Mencari vektor terdekat di Pinecone (cosine similarity)
4. Mengembalikan top-K dokumen paling relevan
"""

import os
from langchain_core.documents import Document
from pinecone import Pinecone
from src.config.settings import (
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_K,
)
from src.ingestion.embedder import SentenceTransformersCustomEmbeddings

# Cache untuk retriever singleton
_retriever_instance = None

class CustomPineconeRetriever:
    """Retriever kustom yang memanggil SDK Pinecone secara langsung."""
    def __init__(self, index, embeddings, k: int):
        self.index = index
        self.embeddings = embeddings
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        # Generate embedding untuk query
        query_vector = self.embeddings.embed_query(query)
        
        # Query Pinecone
        response = self.index.query(
            vector=query_vector,
            top_k=self.k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE
        )
        
        # Convert response matches to Document objects
        docs = []
        for match in response.matches:
            metadata = match.metadata or {}
            text = metadata.get("text", "")
            
            # Mendukung format metadata yang berbeda (contoh: index lama menggunakan 'source' & 'start_page')
            source_file = metadata.get("source_file") or metadata.get("source", "Unknown")
            
            # Ambil halaman, tangani kasus float dari Pinecone (e.g. 468.0)
            page = metadata.get("page")
            if page is None:
                page = metadata.get("start_page", "")
            if isinstance(page, float):
                page = int(page)
                
            doc = Document(
                page_content=text,
                metadata={
                    "source_file": source_file,
                    "page": page
                }
            )
            docs.append(doc)
        return docs


def get_retriever():
    """
    Membuat dan mengembalikan retriever yang terkoneksi ke Pinecone.

    Returns:
        CustomPineconeRetriever: Retriever yang siap digunakan
    """
    global _retriever_instance

    if _retriever_instance is not None:
        return _retriever_instance

    # Validasi API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY belum diatur di file .env")

    try:
        # Inisialisasi model embedding
        embeddings = SentenceTransformersCustomEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        # Koneksi ke Pinecone index
        pc = Pinecone(api_key=api_key)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Buat retriever
        _retriever_instance = CustomPineconeRetriever(
            index=index,
            embeddings=embeddings,
            k=RETRIEVER_K
        )

        return _retriever_instance

    except Exception as e:
        raise ConnectionError(
            f"Gagal membuat retriever dari Pinecone: {str(e)}\n"
            "Pastikan:\n"
            "1. PINECONE_API_KEY sudah benar\n"
            "2. Index sudah dibuat (jalankan ingest.py terlebih dahulu)\n"
            "3. Koneksi internet tersedia"
        )

