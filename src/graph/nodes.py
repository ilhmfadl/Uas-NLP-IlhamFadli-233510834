"""
Modul Nodes LangGraph
=====================

Berisi fungsi-fungsi Node yang digunakan oleh StateGraph NutriGuide AI.
Setiap node menerima state dan mengembalikan update state dalam bentuk dictionary.
"""

import json
from src.graph.state import NutriGuideState
from src.graph.router import classify_query
from src.rag.chain import query_rag
from src.rag.retriever import get_retriever
from src.rag.chain import format_documents, extract_sources
from src.tools.nutrition_lookup import nutrition_lookup
from src.tools.recommendation import recommend_foods
from src.tools.comparison import compare_foods
from src.tools.csv_loader import load_nutrition_data
from src.models.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def classify_query_node(state: NutriGuideState) -> dict:
    """Node 1: Mengklasifikasikan query pengguna."""
    print("🤖 [Node 1: Classify Query] Sedang menentukan kategori pertanyaan...")
    result = classify_query(state)
    print(f"   👉 Tipe query: {result['query_type']}")
    return result


def retrieve_pdf_context_node(state: NutriGuideState) -> dict:
    """Node 2: Mengambil konteks dari PDF referensi menggunakan Pinecone."""
    print("🔍 [Node 2: Retrieve PDF Context] Mengambil dokumen referensi dari Pinecone...")
    query = state["query"]
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        context = format_documents(docs)
        sources = extract_sources(docs)
    except Exception as e:
        print(f"   ⚠️  Gagal retrieve PDF: {str(e)}")
        context = "Konteks PDF tidak tersedia karena masalah koneksi."
        sources = ["Koneksi Pinecone Gagal"]

    return {
        "context": context,
        "sources": sources
    }


def nutrition_lookup_node(state: NutriGuideState) -> dict:
    """Node 3: Mengambil informasi nutrisi dari data CSV."""
    print("📊 [Node 3: Nutrition Lookup] Mencari data nutrisi makanan di Nutrients.csv...")
    query = state["query"]
    llm = get_llm()

    # Ekstrak nama makanan dari query menggunakan LLM
    prompt = ChatPromptTemplate.from_template(
        "Ekstrak HANYA nama bahan makanan atau makanan dari kalimat berikut.\n"
        "Gunakan Bahasa Inggris jika memungkinkan karena dataset kami berbahasa Inggris. Contoh: 'telur' -> 'egg', 'pisang' -> 'banana'.\n"
        "Jika ada beberapa kata, ambil kata benda utamanya saja. Jangan tambahkan kata pengantar lain, hanya nama makanannya saja.\n\n"
        "Pertanyaan: \"{query}\"\n"
        "Nama Makanan:"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        food_name = chain.invoke({"query": query}).strip().lower()
        print(f"   🔎 Nama makanan terekstrak: '{food_name}'")
        result = nutrition_lookup(food_name)
    except Exception as e:
        result = f"⚠️ Gagal mencari data nutrisi: {str(e)}"

    return {
        "csv_result": result,
        "sources": ["Nutrients.csv"]
    }


def recommendation_node(state: NutriGuideState) -> dict:
    """Node 4: Memberikan rekomendasi makanan dari CSV."""
    print("🌟 [Node 4: Recommendation Tool] Menyiapkan daftar rekomendasi dari Nutrients.csv...")
    query = state["query"]
    llm = get_llm()

    # Ekstrak kriteria nutrisi menggunakan LLM
    prompt = ChatPromptTemplate.from_template(
        "Klasifikasikan kriteria nutrisi yang dicari pengguna dari kalimat berikut.\n"
        "Pilih salah satu dari daftar berikut: 'high protein', 'low fat', 'high fiber', 'low calories', 'high calories', 'low carbs'.\n"
        "Hanya kembalikan frasa tersebut, jangan berikan teks lainnya.\n\n"
        "Kalimat: \"{query}\"\n"
        "Kriteria:"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        criteria = chain.invoke({"query": query}).strip().lower()
        print(f"   🔎 Kriteria terekstrak: '{criteria}'")
        result = recommend_foods(criteria)
    except Exception as e:
        result = f"⚠️ Gagal memproses rekomendasi: {str(e)}"

    return {
        "csv_result": result,
        "sources": ["Nutrients.csv"]
    }


def comparison_node(state: NutriGuideState) -> dict:
    """Node 5: Membandingkan kandungan gizi antara dua makanan."""
    print("⚖️ [Node 5: Comparison Tool] Membandingkan dua bahan makanan...")
    query = state["query"]
    llm = get_llm()

    # Ekstrak dua nama makanan menggunakan LLM
    prompt = ChatPromptTemplate.from_template(
        "Ekstrak dua bahan makanan yang ingin dibandingkan dari kalimat berikut.\n"
        "Terjemahkan ke Bahasa Inggris. Contoh: 'telur dan tempe' -> 'egg, tempeh'.\n"
        "Format output harus JSON dengan key 'food1' dan 'food2'. Jangan berikan markdown block, berikan text JSON murni.\n\n"
        "Kalimat: \"{query}\"\n"
        "JSON:"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_json = chain.invoke({"query": query}).strip()
        # Clean JSON string
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:-3].strip()
            
        data = json.loads(raw_json)
        food1 = data.get("food1", "")
        food2 = data.get("food2", "")
        print(f"   🔎 Perbandingan antara: '{food1}' vs '{food2}'")
        result = compare_foods(food1, food2)
    except Exception as e:
        result = f"⚠️ Gagal membandingkan makanan: {str(e)}"

    return {
        "csv_result": result,
        "sources": ["Nutrients.csv"]
    }


def healthy_diet_advisor_node(state: NutriGuideState) -> dict:
    """Node 6: Menggabungkan RAG PDF dan CSV untuk rekomendasi diet sehat."""
    print("🥗 [Node 6: Healthy Diet Advisor] Menggabungkan panduan gizi PDF dan analisis data CSV...")
    query = state["query"]
    
    # 1. Ambil PDF context
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        pdf_context = format_documents(docs)
        pdf_sources = extract_sources(docs)
    except Exception as e:
        pdf_context = "Panduan gizi PDF tidak tersedia."
        pdf_sources = []

    # 2. Ambil data CSV (kita berikan ringkasan kelompok makanan gizi seimbang)
    # Sebagai contoh, kita ambil makanan berprotein tinggi atau berserat tinggi untuk mendukung tips diet
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Berdasarkan pertanyaan berikut, jenis zat gizi apa yang paling relevan dicari? (pilih satu: protein, serat, kalori, atau lemak).\n"
        "Hanya kembalikan satu kata tersebut.\n\n"
        "Pertanyaan: \"{query}\"\n"
        "Gizi:"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        gizi = chain.invoke({"query": query}).strip().lower()
        if gizi == "protein":
            csv_data = recommend_foods("high protein")
        elif gizi == "serat":
            csv_data = recommend_foods("high fiber")
        elif gizi == "lemak":
            csv_data = recommend_foods("low fat")
        else:
            csv_data = recommend_foods("low calories")
    except Exception as e:
        csv_data = "Data CSV tidak tersedia."

    sources = pdf_sources + ["Nutrients.csv"]

    return {
        "context": pdf_context,
        "csv_result": csv_data,
        "sources": sources
    }


def generate_answer_node(state: NutriGuideState) -> dict:
    """Node 7: Menghasilkan jawaban final terintegrasi."""
    print("✍️ [Node 7: Generate Answer] Menyusun jawaban akhir...")
    
    query = state["query"]
    query_type = state["query_type"]
    context = state.get("context", "")
    csv_result = state.get("csv_result", "")
    llm = get_llm()

    # Template prompt dinamis berdasarkan tipe query
    if query_type == "nutritional_qa":
        prompt_text = (
            "Anda adalah NutriGuide AI, asisten gizi tepercaya.\n"
            "Jawablah pertanyaan berdasarkan referensi PDF berikut. Tunjukkan empati dan profesionalisme.\n\n"
            "Konteks PDF:\n{context}\n\n"
            "Pertanyaan: {query}\n\n"
            "Jawaban:"
        )
    elif query_type == "food_lookup":
        prompt_text = (
            "Anda adalah NutriGuide AI. Berdasarkan data nutrisi dari CSV di bawah ini, jelaskan kandungan gizi makanan tersebut secara ramah dan informatif.\n"
            "Tambahkan saran singkat tentang bagaimana mengonsumsinya secara sehat.\n\n"
            "Data Nutrisi:\n{csv_result}\n\n"
            "Pertanyaan: {query}\n\n"
            "Jawaban:"
        )
    elif query_type == "food_recommendation":
        prompt_text = (
            "Anda adalah NutriGuide AI. Jelaskan daftar rekomendasi makanan di bawah ini kepada pengguna.\n"
            "Berikan ringkasan mengapa makanan tersebut direkomendasikan dan bagaimana pengaruhnya terhadap tubuh.\n\n"
            "Daftar Rekomendasi:\n{csv_result}\n\n"
            "Pertanyaan: {query}\n\n"
            "Jawaban:"
        )
    elif query_type == "nutrition_comparison":
        prompt_text = (
            "Anda adalah NutriGuide AI. Terangkan tabel perbandingan nutrisi di bawah ini secara ringkas.\n"
            "Bantu pengguna memahami perbedaan utama antara kedua makanan tersebut.\n\n"
            "Tabel Perbandingan:\n{csv_result}\n\n"
            "Pertanyaan: {query}\n\n"
            "Jawaban:"
        )
    else: # healthy_diet
        prompt_text = (
            "Anda adalah NutriGuide AI. Berikan saran diet dan pola makan sehat yang sangat komprehensif.\n"
            "Gunakan kombinasi panduan gizi dari PDF dan alternatif bahan makanan sehat dari CSV berikut:\n\n"
            "Panduan PDF:\n{context}\n\n"
            "Alternatif Makanan CSV:\n{csv_result}\n\n"
            "Pertanyaan: {query}\n\n"
            "Jawaban:"
        )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    try:
        final_answer = chain.invoke({
            "query": query,
            "context": context,
            "csv_result": csv_result
        })
    except Exception as e:
        final_answer = f"⚠️ Terjadi kesalahan saat memproses jawaban: {str(e)}"

    return {"answer": final_answer}
