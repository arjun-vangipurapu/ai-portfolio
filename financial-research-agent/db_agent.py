# db_agent.py
import sqlite3
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

DB_PATH = "data/metrics.db"

def get_schema() -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema = []
    for table in tables:
        name = table[0]
        cursor.execute(f"PRAGMA table_info({name})")
        cols = cursor.fetchall()
        col_names = [col[1] for col in cols]
        schema.append(f"{name}: {', '.join(col_names)}")
    conn.close()
    return "\n".join(schema)

sql_prompt = PromptTemplate.from_template("""
You are a SQL expert. Generate a SQLite SQL query for the question below.
Return ONLY the SQL query, nothing else. No markdown, no explanation.

Database schema:
{schema}

Companies available: Apple (AAPL), Microsoft (MSFT), Infosys (INFY)

Question: {question}

SQL:
""")

def run_db_agent(question: str) -> str:
    try:
        schema = get_schema()

        # generate SQL
        response = llm.invoke(sql_prompt.format(
            schema=schema,
            question=question
        ))
        sql = response.content.strip()
        # clean markdown if model adds it
        sql = sql.replace("```sql", "").replace("```", "").strip()
        print(f"  🗄️ SQL: {sql}")

        # execute
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            return "No data found for this query."

        return f"DB Results:\n{df.to_string(index=False)}"

    except Exception as e:
        return f"DB Agent error: {e}"

if __name__ == "__main__":
    questions = [
        "What is the market cap and PE ratio of all three companies?",
        "Which company has the highest revenue growth?",
        "Show Apple stock price for the last 5 days"
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(run_db_agent(q))