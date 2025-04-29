"""Handles the interactive CLI loop and memory updates."""
from typing import Any, Dict, Optional
from langchain.memory import ConversationBufferMemory
import os
import json
import platform
import subprocess
import traceback
from visualization.chart_renderer import render_chart


def _update_memory(memory: ConversationBufferMemory, question: str, answer: str, executed_sql: str) -> None:
    """Append the latest interaction (question + answer) to conversation memory."""
    memory.chat_memory.add_user_message(question)
    ai_message = answer
    if executed_sql:
        ai_message += f"\n[ExecutedSQL]: {executed_sql}"
    memory.chat_memory.add_ai_message(ai_message)


def _open_image_in_viewer(image_path: str) -> bool:
    """
    Open an image in the default system viewer.
    Returns True if successful, False otherwise.
    """
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["open", image_path], check=False)
        elif system == "Windows":
            os.startfile(image_path)
        elif system == "Linux":
            subprocess.run(["xdg-open", image_path], check=False)
        else:
            return False
        return True
    except Exception:
        return False


def _process_visualization(chart_json: str) -> None:
    """
    Process a chart JSON specification, render it, and display it.
    """
    print("A visualization has been generated based on your request.")
    
    try:
        # Parse the chart JSON if it's a string
        chart_data = json.loads(chart_json) if isinstance(chart_json, str) else chart_json
        
        # Generate the chart image
        chart_path = render_chart(chart_data)
        
        if chart_path:
            # Get relative path for display
            rel_path = os.path.relpath(chart_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(f"Chart saved to: {rel_path}")
            
            # Display additional information about the chart
            chart_type = chart_data.get("type", "bar")
            print(f"Chart type: {chart_type.capitalize()}")
            
            # Try to open the image with the default image viewer
            if _open_image_in_viewer(chart_path):
                print("Image opened in default viewer.")
            else:
                print(f"Please open the image manually at: {os.path.abspath(chart_path)}")
        else:
            print("Failed to generate chart image.")
    except Exception as e:
        print(f"Error processing chart data: {e}")
        traceback.print_exc()


def _process_agent_response(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Process the agent's response and extract relevant information.
    Returns a dictionary with answer and executed_sql.
    """
    answer = result["answer"]
    executed_sql = result.get("executed_sql", "")
    print(f"Answer: {answer}\n")
    
    # Handle chart specification if present
    if "chart_spec" in result:
        _process_visualization(result["chart_spec"])
    
    return {"answer": answer, "executed_sql": executed_sql}


def _get_user_input() -> Optional[str]:
    """
    Get input from the user.
    Returns None if the user wants to exit, otherwise returns the question.
    """
    try:
        question = input("Question > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return None

    if question.lower() in {"exit", "quit", "q"}:
        return None
    if not question:
        return _get_user_input()
    
    return question


def run_cli(model_name: str, agent_app: Any) -> None:  # noqa: D401
    """
    Run the CLI interface for the SQL agent.
    
    Args:
        model_name: Name of the LLM model being used
        agent_app: The compiled LangGraph agent application
    """
    memory = ConversationBufferMemory(return_messages=True)
    print(
        f"\nAsk questions about the employee database (LangGraph + Memory, {model_name})! "
        "Type 'exit' to quit.\n"
    )

    while True:
        # Get user input
        question = _get_user_input()
        if question is None:
            break

        # Prepare state with chat history
        history_text = "\n".join(
            f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages
        )
        state = {"question": question, "chat_history": history_text}
        
        # Invoke the agent
        result = agent_app.invoke(state)
        
        # Process the agent's response
        response = _process_agent_response(result)
        
        # Update memory
        _update_memory(memory, question, response["answer"], response["executed_sql"])
