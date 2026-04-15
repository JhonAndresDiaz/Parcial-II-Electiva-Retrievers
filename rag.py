from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# ── Configuración ─────────────────────────────────────────────────────────────
DOCS_PATH = "./docs"
DB_PATH   = "./chroma_db"


# ── Modelo Pydantic para análisis estructurado (como en eam_pydantic.py) ──────
class AnalisisNecesidad(BaseModel):
    tipo_producto: str = Field(
        description="Tipo de producto que busca: celular, laptop, tablet, etc."
    )
    presupuesto_max: int = Field(
        description="Presupuesto máximo en pesos colombianos. Usar 0 si no menciona."
    )
    caracteristicas: List[str] = Field(
        description="Lista de características importantes para el usuario"
    )
    uso_principal: str = Field(
        description="Para qué usará el producto principalmente"
    )


# ── 1. CREAR BASE VECTORIAL (SOLO UNA VEZ) ───────────────────────────────────
def create_vector_db():
    chroma_file = os.path.join(DB_PATH, "chroma.sqlite3")
    if os.path.exists(chroma_file):
        print("La base vectorial ya existe. No se recrea.")
        return

    print("Creando base vectorial desde los PDFs...")

    loader = PyPDFDirectoryLoader(DOCS_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Base vectorial creada — {len(chunks)} chunks de {len(docs)} páginas.")


# ── 2. CARGAR CHUNKS PARA BM25 ────────────────────────────────────────────────
def _cargar_chunks():
    loader = PyPDFDirectoryLoader(DOCS_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


# ── 3. CADENA LCEL CON 4 RUNNABLES ───────────────────────────────────────────
#
#  consulta del usuario (str)
#      │
#      ▼  runnable1 — Análisis de necesidad
#      │   LLM + Pydantic (igual a eam_pydantic.py)
#      │   Extrae: tipo, presupuesto, características, uso
#      │   Salida: {"consulta": str, "analisis": AnalisisNecesidad}
#      │
#      ▼  runnable2 — Extracción de criterios
#      │   Arma un query optimizado para la búsqueda RAG
#      │   Salida: {"consulta": str, "analisis": ..., "query": str}
#      │
#      ▼  runnable3 — RAG con EnsembleRetriever
#      │   BM25 (40%) + Chroma (60%) — DIFERENTE a los de clase
#      │   Recupera fragmentos relevantes de los PDFs
#      │   Salida: {"consulta": str, "analisis": str, "contexto": str}
#      │
#      ▼  runnable4 — Recomendación razonada
#          PromptTemplate | LLM | StrOutputParser
#          Salida: str (respuesta final)
#
def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    chunks     = _cargar_chunks()
    llm        = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Chroma — búsqueda semántica sobre los PDFs
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    chroma_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # BM25 — búsqueda léxica sobre los mismos chunks
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # EnsembleRetriever (retriever diferente a los vistos en clase)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )

    # ── RUNNABLE 1 — Análisis de necesidad ───────────────────────────────────
    # Entrada : str
    # Salida  : {"consulta": str, "analisis": AnalisisNecesidad}
    prompt_analisis = PromptTemplate(
        template="Analiza esta consulta de compra y extrae la información clave:\n\n{consulta}",
        input_variables=["consulta"]
    )
    chain_analisis = prompt_analisis | llm.with_structured_output(AnalisisNecesidad)

    def analizar_necesidad(consulta):
        analisis = chain_analisis.invoke({"consulta": consulta})
        return {"consulta": consulta, "analisis": analisis}

    runnable1 = RunnableLambda(analizar_necesidad)

    # ── RUNNABLE 2 — Extracción de criterios ─────────────────────────────────
    # Entrada : {"consulta": str, "analisis": AnalisisNecesidad}
    # Salida  : {"consulta": str, "analisis": ..., "query": str}
    def extraer_criterios(entrada):
        a = entrada["analisis"]
        query = f"{a.tipo_producto} {' '.join(a.caracteristicas)} {a.uso_principal}"
        if a.presupuesto_max > 0:
            query += f" precio {a.presupuesto_max}"
        return {**entrada, "query": query}

    runnable2 = RunnableLambda(extraer_criterios)

    # ── RUNNABLE 3 — RAG con EnsembleRetriever ───────────────────────────────
    # Entrada : {"consulta": str, "analisis": AnalisisNecesidad, "query": str}
    # Salida  : {"consulta": str, "analisis": str, "contexto": str}
    def recuperar_productos(entrada):
        docs = ensemble_retriever.invoke(entrada["query"])
        contexto = "\n\n".join(
            f"[Fragmento {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        a = entrada["analisis"]
        analisis_str = (
            f"Tipo: {a.tipo_producto} | "
            f"Presupuesto: ${a.presupuesto_max:,} COP | "
            f"Características: {', '.join(a.caracteristicas)} | "
            f"Uso: {a.uso_principal}"
        )
        return {
            "consulta": entrada["consulta"],
            "analisis": analisis_str,
            "contexto": contexto
        }

    runnable3 = RunnableLambda(recuperar_productos)

    # ── RUNNABLE 4 — Recomendación razonada ──────────────────────────────────
    # Entrada : {"consulta": str, "analisis": str, "contexto": str}
    # Salida  : str
    prompt_recomendacion = PromptTemplate(
        template="""Eres un asistente de compras experto y amigable.

El cliente pregunta: "{consulta}"

Análisis de su necesidad:
{analisis}

Información encontrada en los PDFs:
{contexto}

Recomienda los mejores 2 o 3 productos que aparecen en la información. Para cada uno explica:
- Por qué se adapta a lo que el cliente necesita
- Pros y contras principales
- Si está dentro del presupuesto indicado

Termina con una recomendación final clara.""",
        input_variables=["consulta", "analisis", "contexto"]
    )

    runnable4 = (
        RunnableLambda(lambda x: prompt_recomendacion.format(**x))
        | llm
        | StrOutputParser()
    )

    # ── CADENA: runnable1 | runnable2 | runnable3 | runnable4 ────────────────
    chain = runnable1 | runnable2 | runnable3 | runnable4

    return chain, ensemble_retriever


# ── Función pública usada por app.py ─────────────────────────────────────────
def query_rag(consulta: str):
    chain, retriever = build_chain()
    respuesta = chain.invoke(consulta)
    docs = retriever.invoke(consulta)
    return respuesta, docs
