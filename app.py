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

---
**Retriever usado**

`EnsembleRetriever`
- **BM25**: busca por palabras clave exactas
- **Chroma**: busca por significado semántico
- Fusión con **Reciprocal Rank Fusion**

""")

    st.divider()
    st.markdown("**Ejemplos de consultas:**")
    st.caption("Quiero un celular barato con buena cámara")
    st.caption("Necesito un laptop para gaming, máximo 4 millones")
    st.caption("¿Qué tablet me sirve para dibujar?")
    st.caption("Busco un smartphone con buena batería para viajes")

st.title("Asistente de Compras Inteligente")
st.caption("Describe lo que necesitas y te recomendaré los mejores productos de nuestros PDFs de smartphones, laptops y tablets.")

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
