"""
Prompts for the visualization agent.
"""
from langchain.prompts import PromptTemplate

VISUALIZATION_PROMPT = PromptTemplate(
    template=(
        "You are a data visualization expert. Generate a chart specification in JSON format based on the following:\n\n"
        "User request: {question}\n\n"
        "Data: {data}\n\n"
        "Create a JSON specification with the proper structure (note: do NOT include any comments in your JSON, ensure it's valid JSON)\n"
        "IMPORTANT: Your response must be ONLY the valid JSON with no comments or explanations.\n"
        "Choose the most appropriate chart type based on the user request and data structure (e.g., bar, pie, line).\n"
        "Make sure all JSON is valid and properly formatted."
    ),
    input_variables=["question", "data"],
)
