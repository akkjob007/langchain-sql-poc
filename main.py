"""
Main CLI entrypoint for LangGraph SQL Q&A agent with memory.
Supports GPT-4o (OpenAI) and Gemini-2.0-flash (Google).
"""
import sys
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from db.setup import DB_URI, init_sample_db
from agents.chat_sql_agent import build_agent
from llm.loader import choose_llm
from cli.runner import run_cli

def main():
    load_dotenv()
    init_sample_db()
    model_name, llm = choose_llm()
    db = SQLDatabase.from_uri(DB_URI)
    agent_app = build_agent(db, llm)
    run_cli(model_name, agent_app)

if __name__ == "__main__":
    main()
