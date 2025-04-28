"""Simple Q&A over SQL demo using LangChain.

Run:
    export GOOGLE_API_KEY=YOUR_KEY
    python main_gemini.py

On first run, creates a sample SQLite database with employee data.
"""
from __future__ import annotations

import os
import pathlib
import re
import sqlite3
from typing import Any, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate

load_dotenv()  # Load GOOGLE_API_KEY from .env if present

DB_PATH = pathlib.Path(__file__).with_name("example.db")
DB_URI = f"sqlite:///{DB_PATH}"

# Sample data for initializing the database
SAMPLE_ROWS: list[tuple[Any, ...]] = [
    (1, "John Doe", 30, "Sales", 70000),
    (2, "Jane Smith", 28, "Engineering", 95000),
    (3, "Alice Johnson", 35, "Marketing", 80000),
    (4, "Bob Williams", 40, "Sales", 88000),
    (5, "Charlie Brown", 25, "Engineering", 65000),
]


def _init_sample_db() -> None:
    """Create a simple table with sample data if DB does not yet exist."""
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


_definitions_loaded = False


def main() -> None:
    _init_sample_db()

    # Initialize components
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    db = SQLDatabase.from_uri(DB_URI)

    # Prompt to generate SQL only (no markdown)
    sql_gen_prompt = PromptTemplate(
        template=(
            "You are a SQLite expert. Given the table schema below and a natural language question, write ONLY the SQL query needed to answer it. "
            "Do NOT include any markdown, comments, or formatting—just the SQL.\n\nSchema:\n{schema}\n\nQuestion: {question}\nSQLQuery:"
        ),
        input_variables=["schema", "question"],
    )

    # Prompt to generate final answer from result
    answer_prompt = PromptTemplate(
        template=(
            "Given the SQL result {result} for the user question '{question}', answer the question in one sentence."
        ),
        input_variables=["result", "question"],
    )

    schema_str = db.get_table_info()

    def _strip_fences(text: str) -> str:
        """Remove markdown code fences if present."""
        text = text.strip()
        # Remove ```blocks```
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
            text = text.rstrip("`")
            text = text.rstrip()
            text = text.rstrip("`")
        text = text.replace("```", "").strip()
        return text

    print("\nAsk questions about the employee database! Type 'exit' to quit.\n")
    while True:
        try:
            query = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        try:
            # 1) Generate SQL
            gen_sql = sql_gen_prompt.format(schema=schema_str, question=query)
            sql_raw = llm.invoke(gen_sql).content.strip()
            sql_clean = _strip_fences(sql_raw)

            # 2) Execute
            result: List[Any] = db.run(sql_clean)  # returns list of tuples

            # 3) Generate final answer
            ans_prompt = answer_prompt.format(result=result, question=query)
            final_answer = llm.invoke(ans_prompt).content.strip()

            print(f"SQL: {sql_clean}\nResult: {result}\nAnswer: {final_answer}\n")
        except Exception as err:  # noqa: BLE001
            print(f"Error: {err}\n")


if __name__ == "__main__":
    main()
