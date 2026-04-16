import streamlit as st
from rag import create_vector_db, query_rag

st.set_page_config(
    page_title="Asistente de Compras",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource(show_spinner="Cargando PDFs y base vectorial...")
def inicializar():
    create_vector_db()

inicializar()

RETRIEVERS_INFO = [
    {
        "nombre": "1. BM25Retriever",
        "tipo": "Léxico",
        "que_es": "Algoritmo de ranking BM25 (Best Match 25), basado en la frecuencia de términos. No usa embeddings ni vectores: pura coincidencia de palabras clave.",
        "para_que": "Cuando el usuario escribe términos exactos que aparecen en los PDFs: marcas (Samsung, Apple), modelos (Galaxy S24) o specs técnicas (200MP, 5000mAh). Rápido y sin API.",
        "ejemplo": """\
retriever = BM25Retriever.from_documents(docs)
retriever.k = 3
resultados = retriever.invoke("celular barato buena cámara")""",
    },
    {
        "nombre": "2. EnsembleRetriever",
        "tipo": "Híbrido",
        "que_es": "Combina múltiples retrievers usando Reciprocal Rank Fusion (RRF). Cada retriever vota por sus documentos y los resultados se fusionan ponderados por peso.",
        "para_que": "Para obtener lo mejor de dos mundos: BM25 captura coincidencias exactas ('nocturna', 'barato') y Chroma captura significado semántico ('económico' = 'bajo precio').",
        "ejemplo": """\
retriever = EnsembleRetriever(
    retrievers=[bm25, chroma],
    weights=[0.4, 0.6]
)
resultados = retriever.invoke("celular barato buena cámara")""",
    },
    {
        "nombre": "3. ParentDocumentRetriever",
        "tipo": "Jerárquico",
        "que_es": "Indexa fragmentos pequeños (hijos) para una búsqueda más precisa, pero devuelve el documento padre (más grande) para dar más contexto al LLM.",
        "para_que": "Fragmenta páginas de PDFs en chunks pequeños para encontrar coincidencias exactas (ej: 'modo nocturno') pero devuelve la página completa para que el LLM tenga todo el contexto.",
        "ejemplo": """\
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,   # chunks de 100 chars
    parent_splitter=parent_splitter  # chunks de 500 chars
)
retriever.add_documents(docs)
resultados = retriever.invoke("celular buena cámara")""",
    },
    {
        "nombre": "4. ContextualCompressionRetriever",
        "tipo": "Post-filtrado con LLM",
        "que_es": "Envuelve a un retriever base y usa un LLM para comprimir cada documento recuperado, extrayendo solo las partes relevantes a la consulta del usuario.",
        "para_que": "Si el usuario pregunta por 'cámara nocturna' y el retriever devuelve un fragmento con precio, batería, RAM, etc., el compresor extrae solo las oraciones sobre la cámara.",
        "ejemplo": """\
compresor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compresor,
    base_retriever=chroma_retriever
)
resultados = retriever.invoke("celular buena cámara nocturna")""",
    },
    {
        "nombre": "5. MultiVectorRetriever",
        "tipo": "Multi-representación",
        "que_es": "Indexa múltiples representaciones de un mismo documento: el texto original, su resumen, y preguntas hipotéticas que ese documento respondería.",
        "para_que": "Genera preguntas hipotéticas a partir de fragmentos de los PDFs. Cuando el usuario hace una pregunta similar, el retriever la conecta con el fragmento correcto aunque no use las mismas palabras del PDF.",
        "ejemplo": """\
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,   # guarda preguntas hipotéticas
    docstore=InMemoryStore(),  # guarda docs originales
    id_key="doc_id"
)
# Indexa preguntas generadas por LLM → apuntan al doc original
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
resultados = retriever.invoke("celular buena cámara")""",
    },
]

with st.sidebar:
    st.header("¿Cómo funciona?")
    st.markdown("""
---
**Retriever usado**

- **BM25**: busca por palabras clave exactas
- **Chroma**: busca por significado semántico
- Fusión con **Reciprocal Rank Fusion**
""")

    st.divider()
    st.subheader("Investigación de Retrievers")
    for r in RETRIEVERS_INFO:
        with st.expander(f"{r['nombre']} — {r['tipo']}"):
            st.markdown("**¿Qué es?**")
            st.write(r["que_es"])
            st.markdown("**¿Para qué sirve?**")
            st.write(r["para_que"])
            st.markdown("**Ejemplo de código:**")
            st.code(r["ejemplo"], language="python")

    st.divider()
    st.markdown("**Acciones rápidas:**")
    ejemplos = [
        "Quiero un celular barato con buena cámara",
        "Necesito un laptop para gaming, máximo 4 millones",
        "¿Qué tablet me sirve para dibujar?",
        "Busco un smartphone con buena batería para viajes",
    ]
    for ejemplo in ejemplos:
        if st.button(ejemplo, use_container_width=True):
            st.session_state["quick_query"] = ejemplo

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

consulta_rapida = st.session_state.pop("quick_query", None)
consulta = consulta_rapida or st.chat_input("¿Qué producto estás buscando?")

if consulta:

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
