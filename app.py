"""
Aplikasi Utama Chatbot Web - NutriGuide AI (Gradio)
===================================================

Modul ini adalah entry point utama untuk menjalankan chatbot interaktif
NutriGuide AI menggunakan antarmuka web Gradio.
"""

import sys
import os
import gradio as gr

# Konfigurasi encoding UTF-8 untuk output terminal Windows agar mendukung emoji
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Tambahkan path proyek agar import berjalan lancar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import validate_environment, PROJECT_ROOT
from src.graph.workflow import run_query, get_workflow_visualization

# Set credentials ke environment secara langsung
os.environ["GROQ_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_3uu7cX_TwA9buTQvgSVDip4xrnsc1RRMdMdPkLwr6nVU6Ja9jHvMVCkCUpntS9P8Vfmo4y"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NutriGuideAI"


def predict(message, history):
    """Fungsi utama untuk memproses query dari UI Gradio."""
    if not message.strip():
        return "", history

    # Jalankan LangGraph Workflow
    final_state = run_query(message)

    answer = final_state.get("answer", "Maaf, tidak dapat memproses jawaban.")
    query_type = final_state.get("query_type", "nutritional_qa").upper()
    sources = final_state.get("sources", [])

    # Format jawaban akhir dengan menyertakan Kategori dan Sumber di bawahnya
    formatted_response = f"**[Kategori: {query_type}]**\n\n{answer}"

    if sources:
        formatted_response += "\n\n---\n**Sumber Referensi:**\n"
        for src in sources:
            formatted_response += f"- {src}\n"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": formatted_response})
    return "", history


# ============================================================
# Konfigurasi Theme dan CSS untuk Gradio 6.x
# ============================================================
theme_config = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="teal",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

custom_css = """
.gradio-container { max-width: 950px !important; margin: auto; }
.header-box { text-align: center; padding: 28px 0 16px 0; }
.header-box h1 { color: #059669; font-weight: 800; font-size: 2.2rem; margin: 0 0 6px 0; }
.header-box p { font-size: 1rem; color: #6b7280; margin: 0; }
.header-line { width: 60px; height: 3px; background: #059669; margin: 14px auto 0 auto; border-radius: 2px; }
.badge-row { display: flex; gap: 10px; margin-top: 12px; justify-content: center; flex-wrap: wrap; }
.badge { background: #f0fdf4; border: 1px solid #d1fae5; border-radius: 6px; padding: 4px 14px; font-size: 0.8rem; color: #065f46; font-weight: 500; }
"""

# ============================================================
# Layout Gradio (Gradio 6.x compatible)
# ============================================================
with gr.Blocks() as demo:
    gr.HTML("""
        <div class="header-box">
            <h1>NutriGuide AI</h1>
            <p>Asisten Cerdas Gizi &amp; Nutrisi Berbasis RAG</p>
            <div class="header-line"></div>
            <div class="badge-row">
                <span class="badge">LangChain</span>
                <span class="badge">LangGraph</span>
                <span class="badge">LangSmith</span>
                <span class="badge">Llama 3.3 70B</span>
                <span class="badge">Pinecone</span>
            </div>
        </div>
    """)

    chatbot = gr.Chatbot(show_label=False, height=500)

    with gr.Group():
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ketik pertanyaan seputar gizi, diet, atau perbandingan nutrisi...",
                show_label=False,
                scale=8,
                container=False
            )
            submit_btn = gr.Button("Kirim", variant="primary", scale=1, min_width=100)
            clear_btn = gr.Button("Bersihkan", variant="secondary", scale=1, min_width=100)

    gr.Examples(
        examples=[
            ["Apa itu konsep gizi seimbang dan pilar utamanya?"],
            ["Berapa jumlah kalori, protein, dan lemak pada telur?"],
            ["Berikan rekomendasi makanan yang tinggi protein"],
            ["Bandingkan oatmeal dan nasi putih"],
            ["Bagaimana menyusun pola makan sehat untuk menurunkan berat badan?"]
        ],
        inputs=msg,
        label="Contoh Pertanyaan Demonstrasi:"
    )

    with gr.Accordion("Detail Arsitektur Sistem", open=False):
        gr.HTML("""
            <div style="background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 5px 0; color: #065f46;">LangGraph Agentic Workflow</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #047857;">Sistem merutekan alur secara cerdas berdasarkan klasifikasi query Anda.</p>
            </div>
        """)

        workflow_diagram = f"""
```mermaid
{get_workflow_visualization()}
```
"""
        gr.Markdown(workflow_diagram)

        gr.HTML("""
            <div style="background-color: #f3f4f6; border-left: 4px solid #4b5563; padding: 15px; border-radius: 5px;">
                <h3 style="margin: 0 0 5px 0; color: #1f2937;">Monitoring LangSmith</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #4b5563;">Setiap eksekusi trace, token usage, dan routing terekam secara real-time di LangSmith Cloud dashboard.</p>
            </div>
        """)

    # Definisikan aksi kirim & bersihkan
    submit_btn.click(predict, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(predict, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    print("\nMenjalankan Gradio Web Server...")
    demo.launch(server_name="127.0.0.1", theme=theme_config, css=custom_css, share=False)
