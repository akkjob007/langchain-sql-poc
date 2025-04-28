"""LangGraph-based SQL Q&A agent using Google Gemini with conversation memory.

Run:
    export GOOGLE_API_KEY=YOUR_KEY  # or present in .env
    python main_gemini_graph.py

This script builds a LangGraph workflow that:
1. Remembers the chat history so follow-up questions have context.
2. Generates SQL (with no markdown fences) to query a local SQLite DB.
3. Executes the SQL and forms a natural-language answer.
"""
from __future__ import annotations

import pathlib
import re
import sqlite3
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Environment & database setup
# ---------------------------------------------------------------------------
load_dotenv()

DB_PATH = pathlib.Path(__file__).with_name("example.db")
DB_URI = f"sqlite:///{DB_PATH}"

SAMPLE_ROWS: list[tuple[Any, ...]] = [
    (1, "John Doe", 30, "Sales", 70000),
    (2, "Jane Smith", 28, "Engineering", 95000),
    (3, "Alice Johnson", 35, "Marketing", 80000),
    (4, "Bob Williams", 40, "Sales", 88000),
    (5, "Charlie Brown", 25, "Engineering", 65000),
]


def _init_sample_db() -> None:
    if DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute(
            """
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                department TEXT,
                salary REAL
            );
            """
        )
        conn.executemany(
            "INSERT INTO employees (id, name, age, department, salary) VALUES (?, ?, ?, ?, ?);",
            SAMPLE_ROWS,
        )
    conn.close()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SQL_GEN_PROMPT = PromptTemplate(
    template=(
        "You are a SQLite expert. Given the table schema and conversation context, "
        "write ONLY the SQL query needed to answer the user's latest question.\n\n"  # noqa: E501
        "Schema:\n{schema}\n\n"  # noqa: E501
        "Chat history:\n{history}\n\n"  # noqa: E501
        "Question: {question}\nSQLQuery:"
    ),
    input_variables=["schema", "history", "question"],
)

ANSWER_PROMPT = PromptTemplate(
    template=(
        "Given the SQL result {result} and the conversation (history below), "
        "answer the user's latest question '{question}' in one sentence.\n\n"  # noqa: E501
        "Chat history:\n{history}"
    ),
    input_variables=["result", "question", "history"],
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class QAState(TypedDict):
    question: str
    chat_history: str  # formatted history string
    sql_query: str
    sql_result: List[Any]
    answer: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = text.rstrip("`")
    text = text.replace("```", "").strip()
    return text


# ---------------------------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------------------------

def build_agent(db: SQLDatabase, llm: ChatGoogleGenerativeAI):
    schema_str = db.get_table_info()

    # --- Node: generate SQL --------------------------------------------------
    def gen_sql(state: QAState) -> QAState:  # noqa: D401
        prompt = SQL_GEN_PROMPT.format(
            schema=schema_str,
            history=state.get("chat_history", ""),
            question=state["question"],
        )
        sql_raw = llm.invoke(prompt).content.strip()
        sql_clean = _strip_fences(sql_raw)
        return {**state, "sql_query": sql_clean}

    # --- Node: execute SQL ---------------------------------------------------
    def exec_sql(state: QAState) -> QAState:  # noqa: D401
        try:
            result = db.run(state["sql_query"])
        except Exception as exc:  # noqa: BLE001
            result = [[f"ERROR: {exc}"]]
        return {**state, "sql_result": result}

    # --- Node: answer --------------------------------------------------------
    def answer_node_fn(state: QAState) -> QAState:  # noqa: D401
        prompt = ANSWER_PROMPT.format(
            result=state["sql_result"],
            question=state["question"],
            history=state.get("chat_history", ""),
        )
        answer = llm.invoke(prompt).content.strip()
        return {**state, "answer": answer}

    # Build graph
    g = StateGraph(QAState)
    g.add_node("gen_sql", gen_sql)
    g.add_node("exec_sql", exec_sql)
    g.add_node("answer_node", answer_node_fn)

    g.set_entry_point("gen_sql")
    g.add_edge("gen_sql", "exec_sql")
    g.add_edge("exec_sql", "answer_node")
    g.add_edge("answer_node", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    _init_sample_db()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    db = SQLDatabase.from_uri(DB_URI)

    # conversation memory
    memory = ConversationBufferMemory(return_messages=True)

    agent_app = build_agent(db, llm)

    print("\nAsk questions about the employee database (LangGraph + Memory)! Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        # Build state with current chat history
        history_text = "\n".join(
            f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages
        )
        state: QAState = {"question": question, "chat_history": history_text}

        result: Dict[str, Any] = agent_app.invoke(state)

        answer = result["answer"]
        print(f"Answer: {answer}\n")

        # update memory
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)


if __name__ == "__main__":
    main()
