from langchain.prompts import PromptTemplate

SQL_GEN_PROMPT = PromptTemplate(
    template=(
        "You are a SQLite expert. Given the table schema and conversation context, "
        "write ONLY the SQL query needed to answer the user's latest question.\n\n"
        "Schema:\n{schema}\n\n"
        "Chat history:\n{history}\n\n"
        "Question: {question}\nSQLQuery:"
    ),
    input_variables=["schema", "history", "question"],
)

ANSWER_PROMPT = PromptTemplate(
    template=(
        "Given the SQL result {result} and the conversation (history below), "
        "answer the user's latest question '{question}' in one sentence.\n\n"
        "Chat history:\n{history}"
    ),
    input_variables=["result", "question", "history"],
)
