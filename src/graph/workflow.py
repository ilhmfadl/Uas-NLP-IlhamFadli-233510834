r"""
Modul Alur Kerja (Workflow) LangGraph
======================================

Modul ini mendefinisikan StateGraph untuk NutriGuide AI.
Alur routing dinamis:

                      [START]
                         │
               [classify_query_node]
                         │
             (Conditional Edge Router)
             /       |        |      \        \
    [nutritional_qa] [lookup] [recom] [compare] [diet]
            │          │        │        │        │
      [retrieve_pdf]  [tool]   [tool]   [tool]  [advise]
            \          |        |        /        /
             \         |        |       /        /
              \        |        |      /        /
                    [generate_answer_node]
                             │
                           [END]
"""

from langgraph.graph import StateGraph, START, END
from src.graph.state import NutriGuideState
from src.graph.nodes import (
    classify_query_node,
    retrieve_pdf_context_node,
    nutrition_lookup_node,
    recommendation_node,
    comparison_node,
    healthy_diet_advisor_node,
    generate_answer_node
)

# Global compiled app cache
_workflow_app = None


def route_query(state: NutriGuideState) -> str:
    """
    Fungsi routing kondisional berdasarkan tipe query yang terdeteksi.

    Args:
        state: State saat ini

    Returns:
        str: Nama node berikutnya yang harus dijalankan
    """
    query_type = state["query_type"]
    
    # Map tipe query ke nama node tujuan
    routing_map = {
        "nutritional_qa": "retrieve_pdf_context_node",
        "food_lookup": "nutrition_lookup_node",
        "food_recommendation": "recommendation_node",
        "nutrition_comparison": "comparison_node",
        "healthy_diet": "healthy_diet_advisor_node"
    }

    # Jika tipe query tidak dikenal, fallback ke RAG PDF
    return routing_map.get(query_type, "retrieve_pdf_context_node")


def create_workflow() -> StateGraph:
    """
    Membangun dan menyusun StateGraph untuk NutriGuide AI.

    Returns:
        StateGraph: Alur kerja graf yang belum di-compile
    """
    # Inisialisasi graph dengan definisi state kita
    workflow = StateGraph(NutriGuideState)

    # 1. Tambahkan semua Node
    workflow.add_node("classify_query_node", classify_query_node)
    workflow.add_node("retrieve_pdf_context_node", retrieve_pdf_context_node)
    workflow.add_node("nutrition_lookup_node", nutrition_lookup_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("comparison_node", comparison_node)
    workflow.add_node("healthy_diet_advisor_node", healthy_diet_advisor_node)
    workflow.add_node("generate_answer_node", generate_answer_node)

    # 2. Hubungkan edge tetap
    # START mengarah langsung ke node klasifikasi query
    workflow.add_edge(START, "classify_query_node")

    # 3. Hubungkan conditional edge (routing otomatis)
    workflow.add_conditional_edges(
        "classify_query_node",
        route_query,
        {
            "retrieve_pdf_context_node": "retrieve_pdf_context_node",
            "nutrition_lookup_node": "nutrition_lookup_node",
            "recommendation_node": "recommendation_node",
            "comparison_node": "comparison_node",
            "healthy_diet_advisor_node": "healthy_diet_advisor_node"
        }
    )

    # 4. Hubungkan semua node pemroses ke generator jawaban akhir
    workflow.add_edge("retrieve_pdf_context_node", "generate_answer_node")
    workflow.add_edge("nutrition_lookup_node", "generate_answer_node")
    workflow.add_edge("recommendation_node", "generate_answer_node")
    workflow.add_edge("comparison_node", "generate_answer_node")
    workflow.add_edge("healthy_diet_advisor_node", "generate_answer_node")

    # 5. Jawaban akhir mengarah ke akhir alur kerja (END)
    workflow.add_edge("generate_answer_node", END)

    return workflow


def get_compiled_app():
    """
    Mengambil atau membuat compiled application dari StateGraph (Singleton).

    Returns:
        CompiledGraph: Aplikasi graf yang siap di-invoke
    """
    global _workflow_app
    if _workflow_app is None:
        workflow = create_workflow()
        _workflow_app = workflow.compile()
    return _workflow_app


def get_workflow_visualization() -> str:
    """
    Mengembalikan string Mermaid untuk visualisasi alur graf.

    Returns:
        str: Kode diagram Mermaid
    """
    app = get_compiled_app()
    # Mengambil graf internal dan menggambarkannya dalam representasi Mermaid
    try:
        return app.get_graph().draw_mermaid()
    except Exception as e:
        return f"Gagal menggambar graf: {str(e)}"


def run_query(query: str) -> dict:
    """
    Menjalankan alur graf lengkap untuk memproses query pengguna.

    Args:
        query: Pertanyaan atau input teks pengguna

    Returns:
        dict: State akhir yang berisi jawaban, tipe query, dan sumber
    """
    app = get_compiled_app()
    
    # Inisialisasi state awal
    initial_state = {
        "query": query,
        "query_type": "",
        "context": "",
        "csv_result": "",
        "answer": "",
        "sources": []
    }

    # Jalankan graf
    final_state = app.invoke(initial_state)
    return final_state
