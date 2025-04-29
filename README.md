# langchain-sql-poc

This project is a demonstration of using LangChain with SQL databases to create a conversational agent. The agent can interact with a SQL database using natural language queries and provides answers based on the database content. It also supports generating visualizations of the data.

## Features
- Supports GPT-4o and Gemini-2.0-flash models.
- Utilizes LangGraph for building a modular agent workflow.
- Conversation memory is managed using LangChain's ConversationBufferMemory.
- **Data Visualization**: Generate charts and graphs from SQL query results.
  - Supports bar charts, pie charts, and line charts
  - Automatically detects visualization requests in user queries
  - Generates and displays chart images

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd langchain-sql-poc
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up environment variables:
   ```bash
   export OPENAI_API_KEY=<your_openai_api_key>
   export GOOGLE_API_KEY=<your_google_api_key>
   ```

## Usage
Run the application using:
```bash
python main.py
```

You will be prompted to select an LLM model and then you can ask questions about the employee database.

### Example Queries
- Basic SQL queries:
  - "What are the salaries of my employees?"
  - "How many employees work in the Engineering department?"
  - "What is the average salary by department?"

- Visualization queries:
  - "Show me a bar chart of employee salaries"
  - "Create a pie chart of employees by department"
  - "Visualize the top 5 highest paid employees"

## Architecture
The application uses a LangGraph-based architecture with the following components:

1. **SQL Agent**: Generates SQL queries based on natural language questions
2. **SQL Executor**: Executes the generated SQL queries against the database
3. **Answer Generator**: Creates natural language answers from SQL results
4. **Visualization Agent**: Generates chart specifications when visualization is requested
5. **Chart Renderer**: Renders chart images from the specifications

## Project Structure
- `agents/`: Contains the SQL and visualization agents
- `cli/`: Command-line interface for the application
- `db/`: Database setup and sample data
- `llm/`: LLM model loading and configuration
- `prompts/`: Prompt templates for the agents
- `visualization/`: Chart rendering and image generation
- `visualizations/`: Generated chart images (not tracked in git)
