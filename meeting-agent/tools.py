from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral")

def load_vectorstore():
    return Chroma(
        persist_directory="./meeting_index",
        embedding_function=embeddings
    )

def semantic_search(query: str) -> str:
    """Search across past meeting transcripts semantically"""
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=3)
    if not docs:
        return "No relevant meetings found."
    results = []
    for doc in docs:
        results.append(
            f"[{doc.metadata['date']} — {doc.metadata['title']}]\n"
            f"Summary: {doc.metadata['summary']}\n"
            f"Decisions: {doc.metadata['decisions']}"
        )
    return "\n\n".join(results)

def action_tracker(owner: str = None) -> str:
    """Get all open action items, optionally filtered by owner"""
    vs = load_vectorstore()
    all_docs = vs.get()
    action_items = []

    for metadata in all_docs["metadatas"]:
        items = json.loads(metadata.get("action_items", "[]"))
        for item in items:
            if item.get("resolved"):
                continue
            if owner and owner.lower() not in item.get("owner", "").lower():
                continue
            item["meeting"] = metadata.get("title")
            item["meeting_date"] = metadata.get("date")
            action_items.append(item)

    if not action_items:
        return "No open action items found."

    lines = []
    for a in action_items:
        lines.append(
            f"• [{a['meeting_date']} — {a['meeting']}] "
            f"{a['task']} — Owner: {a['owner']} "
            f"— Due: {a.get('due', 'not set')}"
        )
    return "\n".join(lines)

def conflict_check(new_decision: str) -> str:
    """Check if a new decision conflicts with past decisions"""
    vs = load_vectorstore()
    docs = vs.similarity_search(new_decision, k=3)

    if not docs:
        return "No past decisions found to compare."

    past_decisions = []
    for doc in docs:
        decisions = json.loads(doc.metadata.get("decisions", "[]"))
        for d in decisions:
            past_decisions.append(f"[{doc.metadata['date']}] {d}")

    conflict_prompt = PromptTemplate.from_template("""
Does the new decision conflict with any past decisions?
Be specific. If no conflict, say "No conflict found."

New decision: {new_decision}

Past decisions:
{past}

Analysis:
""")
    return llm.invoke(conflict_prompt.format(
        new_decision=new_decision,
        past="\n".join(past_decisions)
    ))