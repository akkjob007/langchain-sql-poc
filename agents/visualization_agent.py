"""
Visualization agent for generating chart specifications based on data and user requests.
"""
from typing import Any, Dict, List, TypedDict, Optional
from typing_extensions import Annotated
import json
import re
from prompts.visualization_prompts import VISUALIZATION_PROMPT
from langchain_core.output_parsers import JsonOutputParser

class VisualizationOutput(TypedDict):
    """Chart specification in JSON format."""
    chart_json: Annotated[str, ..., "JSON string containing chart specification"]

class VisualizationRequestOutput(TypedDict):
    """Output for determining if a request is for visualization."""
    is_visualization_request: Annotated[bool, ..., "Whether the request is asking for a visualization"]
    visualization_type: Annotated[str, ..., "The type of visualization requested (e.g., bar, pie, line)"]

def determine_chart_type(request: str, data: List[Dict]) -> str:
    """
    Determine the most appropriate chart type based on the user request and data.
    """
    request_lower = request.lower()
    
    # Check for explicit chart type in request
    if 'pie' in request_lower or 'donut' in request_lower:
        return 'pie'
    elif 'bar' in request_lower:
        return 'bar'
    elif 'line' in request_lower:
        return 'line'
    elif 'scatter' in request_lower:
        return 'scatter'
    elif 'histogram' in request_lower:
        return 'histogram'
    elif 'area' in request_lower:
        return 'area'
    
    # Default chart type based on data structure
    if len(data) <= 1:
        return 'bar'  # Single data point or empty
    elif len(data) <= 10:
        # For small datasets, pie charts work well for categorical data
        # Bar charts work well for numeric comparisons
        return 'bar'
    else:
        # For larger datasets, line or bar charts are usually better
        return 'line'

def is_visualization_request(llm, question: str) -> Dict[str, Any]:
    """
    Use an LLM to determine if a question is requesting a visualization.
    Returns a dictionary with is_visualization_request (bool) and visualization_type (str).
    """
    structured_llm = llm.with_structured_output(VisualizationRequestOutput, method="function_calling")
    
    prompt = """
    Determine if the following user request is asking for a data visualization (chart, graph, plot, etc.).
    
    User request: {question}
    
    Answer with:
    1. is_visualization_request: true if the user is asking for any kind of visual representation of data, false otherwise
    2. visualization_type: the type of visualization requested (e.g., bar, pie, line, scatter) or "general" if not specified
    
    Examples of visualization requests:
    - "Show me a bar chart of sales by region"
    - "Create a pie chart of expenses by category"
    - "Visualize this data"
    - "Can you make a graph of this?"
    - "I'd like to see this information graphically"
    - "Plot the results"
    """
    
    try:
        result = structured_llm.invoke(prompt.format(question=question))
        return {
            "is_visualization_request": result["is_visualization_request"],
            "visualization_type": result["visualization_type"]
        }
    except Exception as e:
        print(f"Error determining if visualization request: {e}")
        # Fallback to simple heuristic if LLM fails
        viz_keywords = ["chart", "graph", "plot", "visualize", "visualization", "pie", "bar", "line"]
        is_viz_request = any(keyword in question.lower() for keyword in viz_keywords)
        return {
            "is_visualization_request": is_viz_request,
            "visualization_type": "general"
        }

def build_visualization_agent(llm) -> Any:
    """Build a visualization agent that generates chart specifications."""
    
    def generate_chart_spec(state: Dict) -> Dict:
        """
        Generate a chart specification based on SQL results and user question.
        """
        # Check if SQL results are available
        if "sql_result" not in state:
            return state
        
        sql_result = state["sql_result"]
        
        # If SQL result is not in expected format, try to extract it
        if isinstance(sql_result, dict) and "result" in sql_result:
            data = sql_result["result"]
        else:
            data = sql_result
            
        # Check if visualization is requested
        question = state["question"]
        viz_request = is_visualization_request(llm, question)
        
        if not viz_request["is_visualization_request"]:
            # No visualization requested
            return state
            
        try:
            # Format the data as JSON string
            data_json = json.dumps(data, indent=2)
            
            # Format the prompt
            prompt = VISUALIZATION_PROMPT.format(
                question=question,
                data=data_json
            )
            
            # Try multiple approaches to get valid JSON
            chart_json = None
            
            # Approach 1: Try with structured output using function calling
            try:
                structured_llm = llm.with_structured_output(VisualizationOutput, method="function_calling")
                result = structured_llm.invoke(prompt)
                chart_json = result["chart_json"]
                print("Successfully generated chart JSON using structured output")
            except Exception as e:
                print(f"Structured output failed: {e}")
                
                # Approach 2: Try with direct JSON parsing
                try:
                    parser = JsonOutputParser()
                    chain = prompt | llm | parser
                    result = chain.invoke({})
                    chart_json = json.dumps(result)
                    print("Successfully generated chart JSON using direct parsing")
                except Exception as e2:
                    print(f"Direct JSON parsing failed: {e2}")
                    
                    # Approach 3: Try with raw completion and manual extraction
                    try:
                        completion = llm.invoke(prompt)
                        content = completion.content
                        
                        # Try to extract JSON from the completion
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            json_str = content.split("```")[1].split("```")[0].strip()
                        else:
                            json_str = content.strip()
                        
                        # Parse the JSON to validate it
                        json.loads(json_str)
                        chart_json = json_str
                        print("Successfully generated chart JSON using raw completion")
                    except Exception as e3:
                        print(f"Raw completion extraction failed: {e3}")
            
            # If we have a valid chart JSON, add it to the state
            if chart_json:
                # Validate JSON
                json.loads(chart_json)
                return {**state, "chart_spec": chart_json}
            
            # If all approaches failed, fall back to a simple chart specification
            raise Exception("All approaches to generate chart JSON failed")
            
        except Exception as e:
            print(f"Failed to generate chart specification: {e}")
            # Fallback to a simple chart specification if LLM fails
            chart_type = viz_request["visualization_type"] if viz_request["visualization_type"] != "general" else determine_chart_type(question, data)
            
            # Extract column names and values
            if len(data) > 0 and isinstance(data[0], dict):
                columns = list(data[0].keys())
                
                if len(columns) >= 2:
                    labels = [str(item.get(columns[0], "")) for item in data]
                    values = [item.get(columns[1], 0) for item in data]
                    
                    fallback_spec = {
                        "type": chart_type,
                        "title": f"{chart_type.capitalize()} Chart of {columns[1]} by {columns[0]}",
                        "labels": labels,
                        "datasets": [
                            {
                                "label": columns[1],
                                "data": values,
                                "backgroundColor": ["#36a2eb", "#ff6384", "#4bc0c0", "#ffcd56", "#9966ff"]
                            }
                        ],
                        "options": {}
                    }
                    
                    return {**state, "chart_spec": json.dumps(fallback_spec)}
            
            # If all else fails, return state unchanged
            return state
    
    return generate_chart_spec
