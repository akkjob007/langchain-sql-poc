"""Simple Q&A over SQL demo using LangChain.

Run:
    export OPENAI_API_KEY=YOUR_KEY
    python main.py

On first run, creates a sample SQLite database with employee data.
"""
from __future__ import annotations

import os
import pathlib
import sqlite3
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


load_dotenv()  # Load OPENAI_API_KEY from .env if present

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
    llm = ChatOpenAI(temperature=0)
    db = SQLDatabase.from_uri(DB_URI)
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

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
            answer = chain.run(query)
            print(f"Answer: {answer}\n")
        except Exception as err:  # noqa: BLE001
            print(f"Error: {err}\n")


if __name__ == "__main__":
    main()
