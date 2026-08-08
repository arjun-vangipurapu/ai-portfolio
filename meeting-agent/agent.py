from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from tools import semantic_search, action_tracker, conflict_check
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

router_prompt = PromptTemplate.from_template("""
You are an agent with three tools. Pick the right one.

Tools:
- semantic_search: questions about discussions, decisions, topics
- action_tracker: questions about tasks, owners, overdue items
- conflict_check: checking if a new decision contradicts past ones

Reply with ONLY one of these exact strings:
semantic_search | action_tracker | conflict_check

Question: {question}
Tool:
""")

synthesize_prompt = PromptTemplate.from_template("""
Answer the user's question using the tool result.
Be concise. Always mention meeting title and date where relevant.

Question: {question}
Tool result: {result}
Answer:
""")

def run_agent():
    print("\n📋 Meeting Intelligence Agent ready.")
    print("─" * 50)
    print("Try asking:")
    print("  → What did we decide about GraphQL?")
    print("  → What action items are open for Sai?")
    print("  → We want to use REST for all services — any conflicts?")
    print("─" * 50)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["q", "quit", "exit"]:
            print("Goodbye.")
            break
        if not question:
            continue

        # Step 1: Route
        response = llm.invoke(
            router_prompt.format(question=question)
        )
        tool_choice = response.content.strip().lower()
        print(f"🔧 Tool: {tool_choice}")

        # Step 2: Execute
        if "action_tracker" in tool_choice:
            known_names = ["sai", "priya", "rahul", "kiran", "ananya"]
            owner = None
            for w in question.split():
                if w.lower() in known_names:
                    owner = w
                    break
            result = action_tracker(owner=owner)
        elif "conflict_check" in tool_choice:
            result = conflict_check(question)
        else:
            result = semantic_search(question)

        # Step 3: Synthesize
        answer = llm.invoke(synthesize_prompt.format(
            question=question,
            result=result
        ))
        print(f"\n🤖 {answer.content}")
        print("─" * 50)

if __name__ == "__main__":
    run_agent()