"""Handles the interactive CLI loop and memory updates."""
from typing import Any
from langchain.memory import ConversationBufferMemory


def _update_memory(memory: ConversationBufferMemory, question: str, answer: str, executed_sql: str) -> None:
    """Append the latest interaction (question + answer) to conversation memory."""
    memory.chat_memory.add_user_message(question)
    ai_message = answer
    if executed_sql:
        ai_message += f"\n[ExecutedSQL]: {executed_sql}"
    memory.chat_memory.add_ai_message(ai_message)


def run_cli(model_name: str, agent_app: Any) -> None:  # noqa: D401
    memory = ConversationBufferMemory(return_messages=True)
    print(
        f"\nAsk questions about the employee database (LangGraph + Memory, {model_name})! "
        "Type 'exit' to quit.\n"
    )

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

        history_text = "\n".join(
            f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages
        )
        state = {"question": question, "chat_history": history_text}
        result = agent_app.invoke(state)

        answer = result["answer"]
        executed_sql = result.get("executed_sql", "")
        print(f"Answer: {answer}\n")

        # memory update
        _update_memory(memory, question, answer, executed_sql)
