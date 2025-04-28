import re
from typing import Any, Dict, List, TypedDict
from langchain_community.utilities import SQLDatabase
from langgraph.graph import END, StateGraph
from prompts.sql_prompts import SQL_GEN_PROMPT, ANSWER_PROMPT
from langchain import hub
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = text.rstrip("`")
    text = text.replace("```", "").strip()
    return text

class QAState(TypedDict):
    question: str
    chat_history: str
    sql_query: str
    executed_sql: str
    sql_result: List[Any]
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def build_agent(db: SQLDatabase, llm) -> Any:
    structured_llm = llm.with_structured_output(QueryOutput)

    def gen_sql(state: QAState) -> QAState:

        prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        })
        sql_query = structured_llm.invoke(prompt)
        return {**state, "sql_query": sql_query}

    def exec_sql(state: QAState) -> QAState:
        executed_sql = state["sql_query"]
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        try:
            result= {"result": execute_query_tool.invoke(executed_sql)}
        except Exception as exc:
            result = [[f"ERROR: {exc}"]]
        return {**state, "sql_result": result, "executed_sql": executed_sql}

    def answer_node_fn(state: QAState) -> QAState:
        prompt = ANSWER_PROMPT.format(
            result=state["sql_result"],
            question=state["question"],
            history=state.get("chat_history", ""),
        )
        answer = llm.invoke(prompt).content.strip()
        return {**state, "answer": answer}

    g = StateGraph(QAState)
    g.add_node("gen_sql", gen_sql)
    g.add_node("exec_sql", exec_sql)
    g.add_node("answer_node", answer_node_fn)
    g.set_entry_point("gen_sql")
    g.add_edge("gen_sql", "exec_sql")
    g.add_edge("exec_sql", "answer_node")
    g.add_edge("answer_node", END)
    return g.compile()
