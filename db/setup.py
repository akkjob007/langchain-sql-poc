import pathlib
import sqlite3
from typing import Any

DB_PATH = pathlib.Path(__file__).parent.parent / "example.db"
DB_URI = f"sqlite:///{DB_PATH}"

SAMPLE_ROWS: list[tuple[Any, ...]] = [
    (1, "John Doe", 30, "Sales", 70000),
    (2, "Jane Smith", 28, "Engineering", 95000),
    (3, "Alice Johnson", 35, "Marketing", 80000),
    (4, "Bob Williams", 40, "Sales", 88000),
    (5, "Charlie Brown", 25, "Engineering", 65000),
]

def init_sample_db() -> None:
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
