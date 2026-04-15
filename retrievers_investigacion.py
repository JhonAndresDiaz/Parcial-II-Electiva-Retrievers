from dotenv import load_dotenv
load_dotenv()

# ============================================================
#  INVESTIGACIÓN DE 5 RETRIEVERS — ASISTENTE DE COMPRAS
#  Fuente de datos: docs/smartphones_guide_updated.pdf
#                   docs/laptops_full.pdf
#                   docs/tablets_full.pdf
#
#  Retrievers VISTOS EN CLASE:
#    • VectorstoreRetriever  → retrievers.py
#    • MultiQueryRetriever   → multi_query_retriever.py
#
#  Retrievers INVESTIGADOS (5 nuevos):
#    1. BM25Retriever
#    2. EnsembleRetriever
#    3. ParentDocumentRetriever
#    4. ContextualCompressionRetriever
#    5. MultiVectorRetriever
# ============================================================

import uuid
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document

# ── Configuración ─────────────────────────────────────────────────────────────
DOCS_PATH = "./docs"
DB_PATH   = "./chroma_db"
PREGUNTA  = "Quiero un celular barato con buena cámara para fotos nocturnas"


# ── Cargar PDFs igual que en los ejercicios del semestre ─────────────────────
def cargar_documentos() -> List[Document]:
    loader   = PyPDFDirectoryLoader(DOCS_PATH)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def _imprimir_resultados(nombre, docs):
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"{'='*60}")
    for i, doc in enumerate(docs[:3], 1):
        linea = doc.page_content.split('\n')[0]   # solo primera línea
        print(f"  [{i}] {linea}")
        print(f"      {doc.page_content[len(linea)+1:200]}...")
    print()


# ════════════════════════════════════════════════════════════
#  RETRIEVER 1 — BM25Retriever
#
#  ¿Qué es?
#    Algoritmo de ranking BM25 (Best Match 25), basado en la
#    frecuencia de términos en los documentos. No usa embeddings
#    ni vectores: pura coincidencia de palabras clave.
#
#  ¿Para qué sirve?
#    Cuando el usuario escribe términos exactos que aparecen
#    en los PDFs: marcas (Samsung, Apple), modelos (Galaxy S24)
#    o specs técnicas (200MP, 5000mAh). Rápido y sin API.
#
#  ¿En qué se diferencia de los vistos en clase?
#    VectorstoreRetriever trabaja con vectores semánticos y
#    necesita embeddings. BM25 no necesita ninguna API externa,
#    es determinista y funciona a nivel de palabra exacta.
# ════════════════════════════════════════════════════════════
def demo_bm25():
    print("\n>>> RETRIEVER 1: BM25Retriever")
    docs = cargar_documentos()

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 3

    resultados = retriever.invoke(PREGUNTA)
    _imprimir_resultados("BM25Retriever — búsqueda léxica sin embeddings", resultados)


# ════════════════════════════════════════════════════════════
#  RETRIEVER 2 — EnsembleRetriever
#
#  ¿Qué es?
#    Combina múltiples retrievers usando Reciprocal Rank
#    Fusion (RRF). Cada retriever vota por sus documentos y
#    los resultados se fusionan ponderados por peso.
#
#  ¿Para qué sirve?
#    Para obtener lo mejor de dos mundos: BM25 captura
#    coincidencias exactas de palabras (ej: "nocturna",
#    "barato") y Chroma captura significado semántico
#    (ej: "económico" = "bajo precio" = "accesible").
#
#  ¿En qué se diferencia de los vistos en clase?
#    No es un solo retriever sino una orquestación. Reduce
#    los puntos ciegos de cada enfoque individual. Es el
#    retriever que usa el sistema RAG principal de este parcial.
# ════════════════════════════════════════════════════════════
def demo_ensemble():
    print("\n>>> RETRIEVER 2: EnsembleRetriever (BM25 + Chroma)")
    docs = cargar_documentos()
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    vectorstore = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)
    chroma = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = EnsembleRetriever(
        retrievers=[bm25, chroma],
        weights=[0.4, 0.6]
    )

    resultados = retriever.invoke(PREGUNTA)
    _imprimir_resultados("EnsembleRetriever — fusión BM25 (40%) + Chroma (60%)", resultados)


# ════════════════════════════════════════════════════════════
#  RETRIEVER 3 — ParentDocumentRetriever
#
#  ¿Qué es?
#    Indexa fragmentos pequeños (hijos) para una búsqueda
#    más precisa, pero devuelve el documento padre (más
#    grande) para dar más contexto al LLM.
#
#  ¿Para qué sirve?
#    Fragmenta las páginas de los PDFs en chunks pequeños
#    para encontrar coincidencias exactas (ej: "modo nocturno")
#    pero devuelve la página completa del PDF para que el LLM
#    tenga todo el contexto del producto.
#
#  ¿En qué se diferencia de los vistos en clase?
#    VectorstoreRetriever devuelve exactamente el chunk
#    indexado. ParentDocumentRetriever devuelve el padre,
#    resolviendo el dilema chunks pequeños vs. contexto rico.
# ════════════════════════════════════════════════════════════
def demo_parent_document():
    print("\n>>> RETRIEVER 3: ParentDocumentRetriever")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    # ParentDocumentRetriever necesita los docs originales (él divide internamente)
    docs = PyPDFDirectoryLoader(DOCS_PATH).load()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10
    )

    vectorstore = Chroma(
        collection_name="parent_doc_retriever",
        embedding_function=embeddings
    )
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents(docs)

    resultados = retriever.invoke(PREGUNTA)
    _imprimir_resultados(
        "ParentDocumentRetriever — busca en chunks pequeños, devuelve producto completo",
        resultados
    )


# ════════════════════════════════════════════════════════════
#  RETRIEVER 4 — ContextualCompressionRetriever
#
#  ¿Qué es?
#    Envuelve a un retriever base y usa un LLM para comprimir
#    cada documento recuperado, extrayendo solo las partes
#    relevantes a la consulta del usuario.
#
#  ¿Para qué sirve?
#    Si el usuario pregunta por "cámara nocturna" y el retriever
#    devuelve un fragmento del PDF con precio, batería, RAM, etc.,
#    el compresor extrae solo las oraciones sobre la cámara.
#    Reduce el ruido en el contexto que recibe el LLM.
#
#  ¿En qué se diferencia de los vistos en clase?
#    Agrega una capa de filtrado post-retrieval con un LLM.
#    Los vistos en clase devuelven el chunk completo tal cual,
#    sin ningún filtrado de relevancia adicional.
# ════════════════════════════════════════════════════════════
def demo_contextual_compression():
    print("\n>>> RETRIEVER 4: ContextualCompressionRetriever")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    vectorstore = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    compresor = LLMChainExtractor.from_llm(llm)

    retriever = ContextualCompressionRetriever(
        base_compressor=compresor,
        base_retriever=base_retriever
    )

    resultados = retriever.invoke(PREGUNTA)
    _imprimir_resultados(
        "ContextualCompressionRetriever — extrae solo info relevante de cada producto",
        resultados
    )


# ════════════════════════════════════════════════════════════
#  RETRIEVER 5 — MultiVectorRetriever
#
#  ¿Qué es?
#    Permite indexar múltiples representaciones de un mismo
#    documento: el texto original, su resumen, y preguntas
#    hipotéticas que ese documento respondería.
#    La búsqueda se hace sobre las representaciones
#    alternativas, pero se devuelve el documento original.
#
#  ¿Para qué sirve?
#    Genera preguntas hipotéticas a partir de fragmentos de
#    los PDFs (ej: "¿Qué celular tiene la mejor cámara nocturna?").
#    Cuando el usuario hace una pregunta similar, el retriever
#    la conecta con el fragmento correcto aunque no use las
#    mismas palabras del PDF.
#
#  ¿En qué se diferencia de los vistos en clase?
#    VectorstoreRetriever indexa un solo vector por documento.
#    MultiVectorRetriever genera N representaciones del mismo
#    documento y busca sobre todas ellas, mejorando el recall.
# ════════════════════════════════════════════════════════════
def demo_multi_vector():
    print("\n>>> RETRIEVER 5: MultiVectorRetriever")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    docs = cargar_documentos()[:6]   # demo rápido con 6 productos

    vectorstore = Chroma(
        collection_name="multi_vector_demo",
        embedding_function=embeddings
    )
    docstore = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key
    )

    print("  Generando preguntas hipotéticas para los primeros 6 fragmentos de los PDFs...")
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    sub_docs = []

    for doc, doc_id in zip(docs, doc_ids):
        prompt = (
            "Genera 2 preguntas cortas que alguien haría antes de comprar este producto. "
            "Solo las preguntas, una por línea.\n\n"
            f"Producto:\n{doc.page_content[:300]}"
        )
        preguntas_texto = llm.invoke(prompt).content
        for pregunta in preguntas_texto.strip().split("\n"):
            if pregunta.strip():
                sub_docs.append(Document(
                    page_content=pregunta.strip(),
                    metadata={id_key: doc_id}
                ))

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    resultados = retriever.invoke(PREGUNTA)
    _imprimir_resultados(
        "MultiVectorRetriever — busca en preguntas hipotéticas, devuelve ficha del producto",
        resultados
    )


# ── Ejecutar todos ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  INVESTIGACIÓN DE 5 RETRIEVERS — ASISTENTE DE COMPRAS")
    print("="*60)
    print(f"\n  Consulta de prueba:\n  '{PREGUNTA}'\n")

    demo_bm25()
    demo_ensemble()
    demo_parent_document()
    demo_contextual_compression()
    demo_multi_vector()

    print("\nInvestigación completada.")
