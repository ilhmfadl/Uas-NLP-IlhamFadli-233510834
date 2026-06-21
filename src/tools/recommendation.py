"""
Modul Food Recommendation Tool
================================

Tool untuk memberikan rekomendasi makanan berdasarkan kriteria tertentu
seperti tinggi protein, rendah lemak, tinggi serat, dll.
Menggunakan data dari Nutrients.csv.
"""

import pandas as pd
from src.tools.csv_loader import load_nutrition_data


def recommend_foods(criteria: str) -> str:
    """
    Memberikan rekomendasi 10 makanan teratas berdasarkan kriteria nutrisi.

    Kriteria yang didukung:
    - tinggi protein / high protein
    - rendah lemak / low fat
    - tinggi serat / high fiber
    - rendah kalori / low calories
    - tinggi kalori / high calories
    - rendah karbohidrat / low carbs / keto

    Args:
        criteria: Kriteria pencarian makanan

    Returns:
        str: Hasil rekomendasi yang diformat
    """
    if not criteria or not criteria.strip():
        return "⚠️ Kriteria rekomendasi tidak boleh kosong."

    df = load_nutrition_data()
    crit = criteria.lower().strip()

    # Tentukan kolom pengurutan dan arah pengurutan
    sort_column = ""
    ascending = False
    criteria_desc = ""

    if "protein" in crit:
        sort_column = "Protein"
        ascending = False
        criteria_desc = "Tinggi Protein"
    elif "lemak" in crit or "fat" in crit:
        if "rendah" in crit or "low" in crit:
            sort_column = "Fat"
            ascending = True
            criteria_desc = "Rendah Lemak"
        else:
            sort_column = "Fat"
            ascending = False
            criteria_desc = "Tinggi Lemak"
    elif "serat" in crit or "fiber" in crit:
        sort_column = "Fiber"
        ascending = False
        criteria_desc = "Tinggi Serat"
    elif "kalori" in crit or "calor" in crit:
        if "rendah" in crit or "low" in crit:
            sort_column = "Calories"
            ascending = True
            criteria_desc = "Rendah Kalori"
        else:
            sort_column = "Calories"
            ascending = False
            criteria_desc = "Tinggi Kalori"
    elif "karbo" in crit or "carb" in crit:
        if "rendah" in crit or "low" in crit:
            sort_column = "Carbs"
            ascending = True
            criteria_desc = "Rendah Karbohidrat"
        else:
            sort_column = "Carbs"
            ascending = False
            criteria_desc = "Tinggi Karbohidrat"
    else:
        # Default fallback
        sort_column = "Protein"
        ascending = False
        criteria_desc = "Tinggi Protein (Default)"

    # Urutkan data berdasarkan kriteria
    # Untuk pencarian rendah kalori/lemak/karbo, kita filter yang nilainya > 0 terlebih dahulu agar tidak memunculkan makanan kosong/porsi 0
    if ascending:
        filtered_df = df[df[sort_column] > 0]
        recommended = filtered_df.sort_values(by=sort_column, ascending=True).head(10)
    else:
        recommended = df.sort_values(by=sort_column, ascending=False).head(10)

    if recommended.empty:
        return f"❌ Tidak ada makanan yang memenuhi kriteria '{criteria_desc}'."

    # Format output
    output_parts = [
        f"🌟 **Top 10 Rekomendasi Makanan - Kategori: {criteria_desc}** 🌟",
        f"{'─' * 60}",
        f"{'Nama Makanan':<30} | {'Porsi':<12} | {'Nilai':<10}"
    ]

    for idx, (_, row) in enumerate(recommended.iterrows(), 1):
        value = row[sort_column]
        unit = "g"
        if sort_column == "Calories":
            unit = "kkal"

        food_name = row['Food']
        if len(food_name) > 28:
            food_name = food_name[:25] + "..."

        output_parts.append(
            f"{idx}. {food_name:<27} | {row['Measure']:<12} | {value:.1f} {unit}"
        )

    output_parts.append(f"{'─' * 60}")
    output_parts.append("\n💡 *Catatan: Data di atas disaring dari Nutrients.csv. Hubungi dokter atau ahli gizi untuk konsultasi diet lebih lanjut.*")

    return "\n".join(output_parts)
