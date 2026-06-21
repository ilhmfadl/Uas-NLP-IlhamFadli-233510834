"""
Modul Inisialisasi LLM (Large Language Model)
==============================================

Modul ini menginisialisasi model LLM yang digunakan oleh NutriGuide AI.

Model: Llama 3.3 70B Versatile
Provider: Groq (inferensi LLM super cepat)

Konfigurasi:
- temperature=0 untuk jawaban deterministik dan konsisten
- Model 70B parameter memiliki kemampuan reasoning yang sangat baik
- Groq menggunakan hardware LPU untuk inferensi < 1 detik
"""

import os
from langchain_groq import ChatGroq
from src.config.settings import LLM_MODEL_NAME, LLM_TEMPERATURE

# Cache untuk singleton pattern
# Hanya membuat satu instance LLM untuk efisiensi memori
_llm_instance = None


def get_llm() -> ChatGroq:
    """
    Mengembalikan instance ChatGroq yang sudah dikonfigurasi.

    Menggunakan singleton pattern: instance dibuat sekali dan di-cache.
    Ini menghindari pembuatan koneksi baru setiap kali LLM dipanggil.

    Returns:
        ChatGroq: Instance LLM yang siap digunakan

    Raises:
        ValueError: Jika GROQ_API_KEY tidak tersedia
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    # Validasi API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY belum diatur. "
            "Silakan atur di file .env atau environment variable.\n"
            "Dapatkan API key gratis di: https://console.groq.com"
        )

    # Buat instance ChatGroq
    _llm_instance = ChatGroq(
        model=LLM_MODEL_NAME,           # llama-3.3-70b-versatile
        temperature=LLM_TEMPERATURE,     # 0 = deterministik
        api_key=api_key,
    )

    return _llm_instance
