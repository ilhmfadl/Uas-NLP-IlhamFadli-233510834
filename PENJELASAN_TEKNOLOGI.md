# Penerapan LangChain, LangGraph, dan LangSmith pada NutriGuide AI

Dokumen ini menjelaskan bagaimana ketiga teknologi utama dari ekosistem LangChain digunakan secara nyata di dalam proyek **NutriGuide AI** agar mudah dipahami, khususnya untuk keperluan demonstrasi UAS.

---

## 1. LangChain 🦜🔗
**Apa itu?** 
LangChain adalah kerangka kerja (framework) utama yang digunakan untuk menghubungkan model AI (seperti Llama 3.3) dengan sumber data eksternal (PDF dan CSV).

**Penerapan di NutriGuide AI:**
- **Pemrosesan Data (Ingestion):** LangChain digunakan untuk membaca file PDF (`PyPDFLoader`), memotong teks menjadi bagian-bagian kecil agar muat di memori AI (`RecursiveCharacterTextSplitter`), dan mengubah teks menjadi angka (vektor) menggunakan *Embeddings*.
- **RAG (Retrieval-Augmented Generation):** LangChain menghubungkan pertanyaan pengguna dengan *Pinecone* (Vector Database) untuk mencari halaman PDF yang paling relevan, lalu mengirimkannya ke Llama 3.3 untuk dirangkum menjadi jawaban.
- **Rantai Proses (Chains):** Menggunakan fitur LCEL (LangChain Expression Language), kita menyusun alur sederhana mulai dari menerima *prompt*, memasukkannya ke LLM, hingga membersihkan output teksnya.

---

## 2. LangGraph 🐙🕸️
**Apa itu?**
Jika LangChain adalah "otak" untuk memproses data, maka LangGraph adalah "manajer lalu lintas" yang cerdas. LangGraph memungkinkan kita membuat sistem agen (*agentic workflow*) yang tidak hanya berjalan lurus/kaku, melainkan bisa mengambil keputusan atau berbelok arah.

**Penerapan di NutriGuide AI:**
- **Routing Cerdas:** Ketika pengguna bertanya, LangGraph akan memasukkan pertanyaan itu ke sebuah "Node Klasifikasi". Sistem kemudian akan memutuskan ke mana pertanyaan ini harus diarahkan.
- **Banyak Jalur (Multi-Flow):** 
  - Jika bertanya tentang teori gizi, LangGraph mengarahkannya ke **Node RAG PDF**.
  - Jika bertanya jumlah kalori telur, LangGraph mengarahkannya ke **Node Lookup CSV**.
  - Jika meminta rekomendasi makanan tinggi protein, arahnya ke **Node Rekomendasi CSV**.
  - Jika meminta menu diet khusus, arahnya ke **Node Healthy Diet (gabungan PDF & CSV)**.
- **Keunggulan:** Berkat LangGraph, chatbot menjadi jauh lebih pintar karena tahu "alat/sumber daya" mana yang harus dipakai sesuai dengan konteks pertanyaan, tanpa perlu mencampuradukkan semua data.

---

## 3. LangSmith 🛠️🔍
**Apa itu?**
LangSmith adalah alat pemantauan (*monitoring* & *observability*) seperti CCTV untuk melihat segala sesuatu yang terjadi di dalam "otak" AI kita.

**Penerapan di NutriGuide AI:**
- **Pelacakan (Tracing):** Setiap kali pengguna mengetik pertanyaan di Gradio, LangSmith merekam seluruh perjalanannya. Kita bisa melihat dari titik mana pertanyaan mulai diproses, node LangGraph mana yang dipilih, hingga dokumen Pinecone apa saja yang diambil.
- **Analisis Kinerja:** Kita dapat melihat berapa lama waktu yang dihabiskan Llama 3.3 untuk membalas pesan (latensi).
- **Pemantauan Biaya & Token:** Lang Smith membantu mencatat berapa banyak *token* yang dihabiskan untuk setiap pertanyaan. Ini sangat berguna di tingkat produksi (production-ready) untuk menghitung efisiensi anggaran.
- Cukup dengan menambahkan 2 baris konfigurasi di file `.env` (`LANGCHAIN_TRACING_V2=true` dan API Key), CCTV ini langsung bekerja di balik layar.

---
**Kesimpulan untuk Dosen Penilai:**
NutriGuide AI bukan sekadar chatbot biasa yang memanggil API ke ChatGPT. Proyek ini mendemonstrasikan **Arsitektur Agentic AI modern**. 
Ia memadukan **LangChain** sebagai jembatan *engine* RAG, **LangGraph** untuk orkestrasi logika pengambilan keputusan secara dinamis (multi-sumber data), serta **LangSmith** untuk membuktikan bahwa seluruh alurnya transparan, termonitor, dan terukur sesuai standar industri rekayasa perangkat lunak (*production-ready*).
