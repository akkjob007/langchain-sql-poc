"""
Prompts for the visualization agent.
"""
from langchain.prompts import PromptTemplate

# Example JSON specification that can be customized
DEFAULT_JSON_EXAMPLE = """{
  "type": "bar",                         // Chart type: bar, pie, line
  "title": "Department Statistics",      // Descriptive title
  "data": {
    "labels": ["Engineering", "Marketing", "Sales", "HR"],
    "datasets": [
      {
        "label": "Average Salary",
        "data": [85000, 95000, 78000, 72000],
        "backgroundColor": [
          "#36a2eb",
          "#ff6384", 
          "#4bc0c0", 
          "#9966ff"
        ],
        "borderColor": [
          "rgb(54, 162, 235)",
          "rgb(255, 99, 132)",
          "rgb(75, 192, 192)",
          "rgb(153, 102, 255)"
        ],
        "borderWidth": 1
      },
      {
        "label": "Number of Employees",
        "data": [12, 8, 10, 3],
        "backgroundColor": "rgba(75, 192, 192, 0.5)",
        "borderColor": "rgb(75, 192, 192)"
      }
    ]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "Department Statistics"
      },
      "legend": {
        "display": true,
        "position": "top"
      }
    }
  }
}"""

VISUALIZATION_PROMPT = PromptTemplate(
    template=(
        "You are a data visualization expert. Generate a concise chart specification in JSON format based on the following:\n\n"
        "User request: {question}\n\n"
        "Data: {data}\n\n"
        "Create a JSON specification with this structure (ensure it's valid JSON with NO comments):\n"
        "```json\n"
        "{json_example}\n"
        "```\n\n"
        "IMPORTANT GUIDELINES:\n"
        "1. Your response must be ONLY valid JSON - no explanations or comments\n"
        "2. Choose the most appropriate chart type based on the data and request\n"
        "3. Use the Chart.js style structure with 'data' property containing 'labels' and 'datasets'\n"
        "4. Keep color specifications simple - use hex colors when possible\n"
        "5. Include only essential properties to reduce token usage\n"
        "6. For multiple datasets, use the same structure with additional objects in the datasets array\n"
        "7. For line charts, always include 'borderColor' property\n"
        "8. For bar/pie charts, always include 'backgroundColor' property\n"
        "9. You can specify colors as single values or arrays matching data points\n"
    ),
    input_variables=["question", "data", "json_example"],
    partial_variables={"json_example": DEFAULT_JSON_EXAMPLE}
)
