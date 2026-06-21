"""
Definisi state untuk LangGraph StateGraph.

State ini menyimpan seluruh informasi yang dibutuhkan
selama proses routing dan pemrosesan query.
"""
from typing import TypedDict


class NutriGuideState(TypedDict):
    """State utama untuk NutriGuide AI workflow."""
    query: str              # Query asli dari pengguna
    query_type: str         # Tipe query setelah klasifikasi
    context: str            # Konteks dari PDF (RAG retrieval)
    csv_result: str         # Hasil dari tools CSV
    answer: str             # Jawaban final
    sources: list[str]      # Daftar sumber referensi
