# langchain-sql-poc

This project is a demonstration of using LangChain with SQL databases to create a conversational agent. The agent can interact with a SQL database using natural language queries and provides answers based on the database content.

## Features
- Supports GPT-4o and Gemini-2.0-flash models.
- Utilizes LangChain's SQLDatabaseChain for database interactions.
- Conversation memory is managed using LangChain's ConversationBufferMemory.

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

## License
This project is licensed under the MIT License.