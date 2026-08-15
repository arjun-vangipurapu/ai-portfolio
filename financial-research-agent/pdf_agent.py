# pdf_agent.py
import os
import pymupdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import re

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    print(f"⚠️ Reranker load failed: {e}. Using similarity search only.")
    reranker = None

TRANSCRIPT_DIR = "data/transcripts"
CHROMA_DIR = "data/chroma_transcripts"

def load_pdfs() -> list[Document]:
    docs = []
    for filename in os.listdir(TRANSCRIPT_DIR):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(TRANSCRIPT_DIR, filename)
        pdf = pymupdf.open(path)
        company = filename.split("_")[0]
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if len(text.strip()) < 50:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "company": company,
                    "page": page_num + 1
                }
            ))
        print(f"  ✅ Loaded: {filename} ({len(pdf)} pages)")
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        # chunk by paragraph with smaller size
        paragraphs = [p.strip() for p in doc.page_content.split("\n\n") 
                     if len(p.strip()) > 50]  # lower threshold from 100 to 50
        for para in paragraphs:
            chunks.append(Document(
                page_content=para,
                metadata=doc.metadata
            ))
    return chunks

def build_index():
    print("Building PDF index...")
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    print(f"  Total chunks: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"  ✅ ChromaDB index built")
    return vectorstore, chunks

# module-level cache
_vectorstore = None
_chunks = None

def load_index():
    global _vectorstore, _chunks
    
    # return cached if already loaded
    if _vectorstore is not None and _chunks is not None:
        print("  ⚡ Using cached index")
        return _vectorstore, _chunks
    
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("  📂 Loading existing index...")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        docs = load_pdfs()
        _chunks = chunk_documents(docs)
        return _vectorstore, _chunks
    
    _vectorstore, _chunks = build_index()
    return _vectorstore, _chunks

def hybrid_search(query: str, vectorstore, chunks: list[Document], k: int = 10) -> list[Document]:
    # semantic search
    semantic_results = vectorstore.similarity_search(query, k=k)

    # BM25 keyword search
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_results = [chunks[i] for i in top_bm25_indices]

    # merge + deduplicate
    seen = set()
    merged = []
    for doc in semantic_results + bm25_results:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    # rerank
    if reranker is not None and len(merged) > 3:
        pairs = [[query, doc.page_content] for doc in merged]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, merged), reverse=True)
        merged = [doc for _, doc in ranked[:3]]
    else:
        merged = merged[:3]
    return merged

answer_prompt = PromptTemplate.from_template("""
You are a financial analyst. Answer the question using ONLY the transcript excerpts below.
If searching for strategy or plans, also look for related terms like product names,
revenue segments, or executive quotes about future direction.
Always mention the company name and source page.
If the answer is not in the excerpts, say "Not found in transcripts."

Excerpts:
{context}

Question: {question}

Answer:
""")

def run_pdf_agent(question: str) -> str:
    try:
        vectorstore, chunks = load_index()
        expanded_query = question
        if any(word in question.lower() for word in [
            "strategy", "plan", "future", "outlook", 
            "guidance", "match", "revenue"
        ]):
            expanded_query = f"{question} quarterly revenue record billion September quarter results"

        results = hybrid_search(expanded_query, vectorstore, chunks)

        if not results:
            return "No relevant content found in transcripts."

        context = "\n\n---\n\n".join([
            f"[{doc.metadata['source']} p.{doc.metadata['page']}]\n{doc.page_content[:800]}"
            for doc in results
        ])

        response = llm.invoke(answer_prompt.format(
            context=context,
            question=question
        ))
        return response.content

    except Exception as e:
        return f"PDF Agent error: {e}"

if __name__ == "__main__":
    questions = [
        "What did Tim Cook say about iPhone revenue?",
        "What was Microsoft's AI strategy discussed in the earnings call?",
        "What guidance did Infosys give for revenue growth?"
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(run_pdf_agent(q))
        print("-" * 50)