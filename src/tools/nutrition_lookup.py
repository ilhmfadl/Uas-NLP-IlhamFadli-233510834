"""
Modul Nutrition Lookup Tool
============================

Tool untuk mencari informasi nutrisi makanan tertentu
berdasarkan data dari Nutrients.csv.

Contoh penggunaan:
- "Berapa kalori pisang?"
- "Berapa protein telur?"
- "Kandungan nutrisi alpukat?"
"""

from src.tools.csv_loader import search_food, load_nutrition_data


def nutrition_lookup(food_name: str) -> str:
    """
    Mencari dan menampilkan informasi nutrisi suatu makanan.

    Args:
        food_name: Nama makanan yang dicari

    Returns:
        str: Informasi nutrisi dalam format tabel yang mudah dibaca.
             Jika tidak ditemukan, mengembalikan pesan saran.
    """
    if not food_name or not food_name.strip():
        return "⚠️ Nama makanan tidak boleh kosong."

    # Cari makanan di dataset
    results = search_food(food_name.strip())

    if results.empty:
        # Cari makanan yang mirip untuk saran
        df = load_nutrition_data()
        all_foods = df['Food'].str.lower().tolist()

        # Cari yang mengandung sebagian kata
        words = food_name.lower().split()
        suggestions = []
        for word in words:
            if len(word) > 2:  # Abaikan kata pendek
                for food in all_foods:
                    if word in food and food not in suggestions:
                        suggestions.append(food)
                        if len(suggestions) >= 5:
                            break

        result = f"❌ Makanan '{food_name}' tidak ditemukan dalam database.\n"
        if suggestions:
            result += "\n💡 Mungkin yang Anda maksud:\n"
            for s in suggestions[:5]:
                result += f"   - {s.title()}\n"

        return result

    # Format hasil sebagai tabel
    output_parts = []

    for _, row in results.iterrows():
        output_parts.append(
            f"🍽️  {row['Food']}\n"
            f"{'─' * 40}\n"
            f"  📏 Porsi       : {row['Measure']}\n"
            f"  ⚖️  Berat       : {row['Grams']:.0f} gram\n"
            f"  🔥 Kalori      : {row['Calories']:.0f} kkal\n"
            f"  💪 Protein     : {row['Protein']:.1f} gram\n"
            f"  🧈 Lemak       : {row['Fat']:.1f} gram\n"
            f"  🧈 Lemak Jenuh : {row['Sat.Fat']:.1f} gram\n"
            f"  🌾 Karbohidrat : {row['Carbs']:.1f} gram\n"
            f"  🥦 Serat       : {row['Fiber']:.1f} gram\n"
            f"  📂 Kategori    : {row['Category']}"
        )

    header = f"📊 Ditemukan {len(results)} hasil untuk '{food_name}':\n\n"
    return header + "\n\n".join(output_parts)
