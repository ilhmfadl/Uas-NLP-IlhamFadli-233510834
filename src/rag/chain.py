"""
Modul RAG Chain
===============

Modul ini membangun pipeline RAG (Retrieval-Augmented Generation)
menggunakan LCEL (LangChain Expression Language).

Pipeline:
Query → Retriever → Context Formatting → Prompt → LLM → Answer
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.retriever import get_retriever
from src.models.llm import get_llm

RAG_PROMPT_TEMPLATE = """Anda adalah NutriGuide AI, asisten nutrisi dan gizi yang ahli.

Jawab pertanyaan berikut HANYA berdasarkan konteks yang diberikan.
Jika informasi tidak tersedia dalam konteks, katakan dengan jujur bahwa
informasi tersebut tidak ditemukan dalam dokumen referensi.

Berikan jawaban yang:
- Lengkap dan informatif
- Mudah dipahami
- Dalam Bahasa Indonesia
- Menyebutkan sumber jika memungkinkan

Konteks dari dokumen referensi:
{context}

Pertanyaan: {question}

Jawaban:"""


def format_documents(docs: list) -> str:
    """
    Memformat daftar dokumen menjadi string konteks.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', '')
        
        # Format halaman (tambah 1 karena metadata dari PyPDFLoader 0-indexed)
        page_str = "N/A"
        if page != '':
            try:
                page_str = str(int(page) + 1)
            except (ValueError, TypeError):
                page_str = str(page)
                
        formatted.append(
            f"[Dokumen {i} | Sumber: {source} | Halaman: {page_str}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def extract_sources(docs: list) -> list:
    """
    Mengekstrak metadata sumber dari dokumen yang di-retrieve.

    Args:
        docs: Daftar objek Document dari retriever

    Returns:
        list: Daftar string berisi informasi sumber
    """
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')
        page = doc.metadata.get('page', '')
        
        source_str = f"{source}"
        if page != '':
            try:
                source_str += f" (Halaman {int(page) + 1})"
            except (ValueError, TypeError):
                source_str += f" (Halaman {page})"

        if source_str not in seen:
            sources.append(source_str)
            seen.add(source_str)
    return sources


def query_rag(question: str) -> dict:
    """
    Menjalankan pipeline RAG lengkap untuk menjawab pertanyaan.

    Proses:
    1. Retriever mencari dokumen relevan di Pinecone
    2. Dokumen diformat menjadi konteks
    3. Prompt diisi dengan konteks dan pertanyaan
    4. LLM menghasilkan jawaban berdasarkan konteks
    5. Metadata sumber diekstrak

    Args:
        question: Pertanyaan dari pengguna

    Returns:
        dict: {
            'answer': str (jawaban dari LLM),
            'sources': list (daftar sumber dokumen),
            'context': str (konteks yang digunakan)
        }
    """
    # Dapatkan retriever dan LLM
    retriever = get_retriever()
    llm = get_llm()

    # Buat prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Buat LCEL chain: prompt → LLM → string output
    chain = prompt | llm | StrOutputParser()

    # Retrieve dokumen relevan
    docs = retriever.invoke(question)

    # Format konteks dan ekstrak sumber
    context = format_documents(docs)
    sources = extract_sources(docs)

    # Generate jawaban menggunakan LLM
    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }
