"""
Modul Router LangGraph
=====================

Modul ini bertanggung jawab untuk mengklasifikasikan query dari pengguna
ke salah satu dari 5 kategori menggunakan Llama 3.3 70B Versatile.
"""

import os
from src.models.llm import get_llm
from src.graph.state import NutriGuideState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def classify_query(state: NutriGuideState) -> dict:
    """
    Mengklasifikasikan query pengguna ke salah satu dari kategori berikut:
    1. 'nutritional_qa'
    2. 'food_lookup'
    3. 'food_recommendation'
    4. 'nutrition_comparison'
    5. 'healthy_diet'

    Args:
        state: State saat ini

    Returns:
        dict: State update dengan query_type terisi
    """
    query = state["query"]
    llm = get_llm()

    prompt_text = """Anda adalah router AI yang bertugas mengklasifikasikan pertanyaan pengguna ke dalam salah satu dari 5 kategori berikut.

Kategori yang tersedia:
1. `nutritional_qa`: Pertanyaan nutrisi, gizi, kesehatan secara umum. Jawaban akan diambil dari buku/dokumen PDF.
   Contoh: "Apa itu gizi seimbang?", "Apa manfaat serat?", "Apa dampak kelebihan konsumsi gula?"
2. `food_lookup`: Mencari tahu detail nutrisi spesifik suatu makanan.
   Contoh: "Berapa kalori pisang?", "Berapa kandungan protein telur?", "Berapa lemak alpukat?"
3. `food_recommendation`: Meminta rekomendasi makanan berdasarkan kriteria gizi tertentu.
   Contoh: "Saya ingin makanan tinggi protein", "Rekomendasi makanan rendah lemak", "Makanan tinggi serat"
4. `nutrition_comparison`: Membandingkan dua jenis makanan.
   Contoh: "Bandingkan telur dan tempe", "Lebih bagus nasi putih atau oatmeal?"
5. `healthy_diet`: Meminta saran pola makan, rencana diet sehat, menurunkan/menaikkan berat badan. Memerlukan integrasi PDF RAG + CSV data.
   Contoh: "Bagaimana cara sehat menurunkan berat badan?", "Pola makan sehat untuk menambah berat badan"

PENTING: Output Anda harus HANYA berupa nama kategori saja: `nutritional_qa`, `food_lookup`, `food_recommendation`, `nutrition_comparison`, atau `healthy_diet`. Jangan ada teks, penjelasan, spasi, atau karakter lain.

Pertanyaan Pengguna: "{query}"

Kategori:"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    try:
        raw_category = chain.invoke({"query": query})
        classified_type = raw_category.strip().lower().replace("`", "").replace("'", "").replace("\"", "")

        # Fallback jika klasifikasi tidak sesuai daftar
        valid_types = ['nutritional_qa', 'food_lookup', 'food_recommendation', 'nutrition_comparison', 'healthy_diet']
        if classified_type not in valid_types:
            # Cari substring kecocokan terdekat
            for t in valid_types:
                if t in classified_type:
                    classified_type = t
                    break
            else:
                classified_type = 'nutritional_qa' # Default fallback
    except Exception as e:
        print(f"⚠️  Gagal klasifikasi: {str(e)}")
        classified_type = 'nutritional_qa'

    return {"query_type": classified_type}
