"""
Modul Nutrition Comparison Tool
================================

Tool untuk membandingkan kandungan gizi antara dua makanan
berdasarkan data dari Nutrients.csv.

Contoh penggunaan:
- "Bandingkan telur dan tempe"
- "Perbandingan nasi putih dan oatmeal"
"""

from src.tools.csv_loader import search_food, load_nutrition_data
from src.models.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def compare_foods(food1: str, food2: str) -> str:
    """
    Membandingkan nilai gizi dari dua jenis makanan.

    Args:
        food1: Nama makanan pertama
        food2: Nama makanan kedua

    Returns:
        str: Tabel perbandingan nutrisi dan kesimpulan dari LLM
    """
    if not food1 or not food2:
        return "⚠️ Masukkan dua nama makanan untuk dibandingkan."

    res1 = search_food(food1.strip())
    res2 = search_food(food2.strip())

    if res1.empty:
        return f"❌ Makanan pertama '{food1}' tidak ditemukan dalam database."
    if res2.empty:
        return f"❌ Makanan kedua '{food2}' tidak ditemukan dalam database."

    # Ambil baris pertama dari hasil pencarian masing-masing
    row1 = res1.iloc[0]
    row2 = res2.iloc[0]

    # Buat tabel perbandingan dalam format Markdown/Text
    val_grams1 = f"{row1['Grams']:.0f} g"
    val_grams2 = f"{row2['Grams']:.0f} g"
    val_cal1 = f"{row1['Calories']:.0f} kkal"
    val_cal2 = f"{row2['Calories']:.0f} kkal"
    val_prot1 = f"{row1['Protein']:.1f} g"
    val_prot2 = f"{row2['Protein']:.1f} g"
    val_fat1 = f"{row1['Fat']:.1f} g"
    val_fat2 = f"{row2['Fat']:.1f} g"
    val_carbs1 = f"{row1['Carbs']:.1f} g"
    val_carbs2 = f"{row2['Carbs']:.1f} g"
    val_fiber1 = f"{row1['Fiber']:.1f} g"
    val_fiber2 = f"{row2['Fiber']:.1f} g"

    comparison_table = (
        f"📊 **Tabel Perbandingan Kandungan Gizi**\n"
        f"{'─' * 60}\n"
        f"{'Nutrisi':<20} | {row1['Food'][:18]:<18} | {row2['Food'][:18]:<18}\n"
        f"{'─' * 60}\n"
        f"{'Porsi':<20} | {row1['Measure']:<18} | {row2['Measure']:<18}\n"
        f"{'Berat':<20} | {val_grams1:<18} | {val_grams2:<18}\n"
        f"{'Kalori':<20} | {val_cal1:<18} | {val_cal2:<18}\n"
        f"{'Protein':<20} | {val_prot1:<18} | {val_prot2:<18}\n"
        f"{'Lemak':<20} | {val_fat1:<18} | {val_fat2:<18}\n"
        f"{'Karbohidrat':<20} | {val_carbs1:<18} | {val_carbs2:<18}\n"
        f"{'Serat':<20} | {val_fiber1:<18} | {val_fiber2:<18}\n"
        f"{'─' * 60}\n"
    )

    # Gunakan LLM untuk memberikan kesimpulan yang cerdas dan relevan
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Anda adalah pakar nutrisi. Berikan kesimpulan singkat dan analisis gizi berdasakan perbandingan berikut:\n\n"
        "Makanan A: {food_a} (Kalori: {cal_a} kkal, Protein: {prot_a}g, Lemak: {fat_a}g, Karbohidrat: {carb_a}g, Serat: {fib_a}g)\n"
        "Makanan B: {food_b} (Kalori: {cal_b} kkal, Protein: {prot_b}g, Lemak: {fat_b}g, Karbohidrat: {carb_b}g, Serat: {fib_b}g)\n\n"
        "Berikan kesimpulan tentang makanan mana yang lebih baik untuk tujuan tertentu (misal: membentuk otot, menurunkan berat badan, atau menambah serat)."
        "Gunakan Bahasa Indonesia yang ramah, ringkas, dan profesional. Jangan memberikan markdown yang terlalu panjang, cukup 3-4 kalimat."
    )

    chain = prompt | llm | StrOutputParser()

    try:
        conclusion = chain.invoke({
            "food_a": row1['Food'], "cal_a": row1['Calories'], "prot_a": row1['Protein'], "fat_a": row1['Fat'], "carb_a": row1['Carbs'], "fib_a": row1['Fiber'],
            "food_b": row2['Food'], "cal_b": row2['Calories'], "prot_b": row2['Protein'], "fat_b": row2['Fat'], "carb_b": row2['Carbs'], "fib_b": row2['Fiber']
        })
    except Exception as e:
        conclusion = f"⚠️ Gagal menghasilkan kesimpulan otomatis dari LLM: {str(e)}"

    return comparison_table + "\n💡 **Analisis Ahli Gizi (LLM):**\n" + conclusion
