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

VALID_TOOLS = ["semantic_search", "action_tracker", "conflict_check"]
KNOWN_NAMES = ["sai", "priya", "rahul", "kiran", "ananya"]

router_prompt = PromptTemplate.from_template("""
You are an agent with three tools. Pick one or more tools needed.
Reply with a comma-separated list only. No explanation.

Tools:
- semantic_search: questions about discussions, decisions, topics
- action_tracker: questions about tasks, owners, overdue items
- conflict_check: checking if a new decision contradicts past ones

Examples:
"What did we decide about GraphQL?" → semantic_search
"What is overdue for Sai?" → action_tracker
"We want REST — any conflicts?" → conflict_check
"What did we decide and what's pending for Sai?" → semantic_search,action_tracker

Question: {question}
Tools:
""")

synthesize_prompt = PromptTemplate.from_template("""
Answer the user's question using the tool results below.
Be concise. Mention meeting title and date where relevant.
End with "Confidence: High/Medium/Low" based on how directly
the results answer the question.

Question: {question}
Tool results:
{result}
Answer:
""")

def extract_owner(question: str) -> str:
    for word in question.split():
        if word.lower() in KNOWN_NAMES:
            return word
    return None

def execute_tool(tool: str, question: str) -> str:
    if tool == "action_tracker":
        return action_tracker(owner=extract_owner(question))
    elif tool == "conflict_check":
        return conflict_check(question)
    elif tool == "semantic_search":
        return semantic_search(question)
    return ""

def run_agent():
    print("\n📋 Meeting Intelligence Agent ready.")
    print("─" * 50)
    print("Try asking:")
    print("  → What did we decide about GraphQL?")
    print("  → What action items are open for Sai?")
    print("  → We want to use REST for all services — any conflicts?")
    print("  → What did we decide and what's pending for Sai?")
    print("─" * 50)

    total_tokens = 0

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ["q", "quit", "exit"]:
            print(f"\n💰 Total tokens used this session: {total_tokens}")
            print("Goodbye.")
            break
        if not question:
            continue

        # Step 1: Route — support multiple tools
        response = llm.invoke(router_prompt.format(question=question))
        raw_tools = response.content.strip().lower()

        # track tokens
        usage = response.response_metadata.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)

        # parse + validate tools
        tools_needed = [
            t.strip() for t in raw_tools.split(",")
            if t.strip() in VALID_TOOLS
        ]
        if not tools_needed:
            print(f"⚠️ Unexpected routing output: '{raw_tools}' — defaulting to semantic_search")
            tools_needed = ["semantic_search"]

        print(f"🔧 Tools: {', '.join(tools_needed)}")

        # Step 2: Execute all tools
        results = []
        for tool in tools_needed:
            tool_result = execute_tool(tool, question)
            results.append(f"[{tool}]\n{tool_result}")

        combined_result = "\n\n".join(results)

        # Step 3: Synthesize
        answer_response = llm.invoke(synthesize_prompt.format(
            question=question,
            result=combined_result
        ))

        usage = answer_response.response_metadata.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)

        print(f"\n🤖 {answer_response.content}")
        print(f"💰 Tokens this query: {usage.get('total_tokens', 0)} | Session total: {total_tokens}")
        print("─" * 50)

if __name__ == "__main__":
    run_agent()