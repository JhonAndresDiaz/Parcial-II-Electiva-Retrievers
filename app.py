import streamlit as st
from rag import create_vector_db, query_rag

st.set_page_config(
    page_title="Asistente de Compras",
    page_icon="🛒",
    layout="wide"
)

# ── Inicializar base vectorial una sola vez ───────────────────────────────────
@st.cache_resource(show_spinner="Cargando PDFs y base vectorial...")
def inicializar():
    create_vector_db()

inicializar()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("¿Cómo funciona?")
    st.markdown("""
**Cadena LCEL — 4 Runnables**

```
Tu consulta (str)
    │
    ▼ Runnable 1
    Análisis de necesidad
    LLM + Pydantic
    → tipo, presupuesto,
      características, uso
    │
    ▼ Runnable 2
    Extracción de criterios
    → query optimizado
      para búsqueda RAG
    │
    ▼ Runnable 3
    RAG — EnsembleRetriever
    BM25 (40%) + Chroma (60%)
    → fragmentos de los PDFs
    │
    ▼ Runnable 4
    Recomendación razonada
    Gemini 2.5 Flash
    → recomendación final
```

---
**Retriever usado**

`EnsembleRetriever`
- **BM25**: busca por palabras clave exactas
- **Chroma**: busca por significado semántico
- Fusión con **Reciprocal Rank Fusion**

Diferente a los vistos en clase:
- VectorstoreRetriever
- MultiQueryRetriever

---
**Fuentes de conocimiento**
- 📱 smartphones_guide_updated.pdf
- 💻 laptops_full.pdf
- 📟 tablets_full.pdf
""")

    st.divider()
    st.markdown("**Ejemplos de consultas:**")
    st.caption("Quiero un celular barato con buena cámara")
    st.caption("Necesito un laptop para gaming, máximo 4 millones")
    st.caption("¿Qué tablet me sirve para dibujar?")
    st.caption("Busco un smartphone con buena batería para viajes")

# ── Título principal ──────────────────────────────────────────────────────────
st.title("Asistente de Compras Inteligente")
st.caption("Describe lo que necesitas y te recomendaré los mejores productos de nuestros PDFs de smartphones, laptops y tablets.")

# ── Historial de mensajes ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "docs" in msg:
            with st.expander("Fragmentos encontrados en los PDFs"):
                for i, doc in enumerate(msg["docs"], 1):
                    st.write(f"**Producto {i}:**")
                    st.write(doc.page_content[:300])

# ── Entrada del usuario ───────────────────────────────────────────────────────
if consulta := st.chat_input("¿Qué producto estás buscando?"):

    st.session_state.messages.append({"role": "user", "content": consulta})

    with st.chat_message("user"):
        st.write(consulta)

    with st.chat_message("assistant"):
        with st.spinner("Analizando tu necesidad y buscando en el catálogo..."):
            respuesta, docs = query_rag(consulta)

        st.write(respuesta)

        with st.expander("Fragmentos encontrados en los PDFs"):
            for i, doc in enumerate(docs, 1):
                st.write(f"**Producto {i}:**")
                st.write(doc.page_content[:300])

    st.session_state.messages.append({
        "role": "assistant",
        "content": respuesta,
        "docs": docs
    })
