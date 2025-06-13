import os
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.utilities.python import PythonREPL
from langchain_community.tools import Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Ensure OpenAI API Key is set in environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Load LLM
llm = OpenAI(temperature=0)

# Load the calculator tool
python_repl = PythonREPL()
python_tool = Tool(
    name="Python REPL",
    description="A Python shell. Use this to execute python commands.",
    func=python_repl.run
)

# Initialize the agent with tools
agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print("🤖 Ask a math question! Type 'exit' to quit.")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    try:
        response = agent.run(query)
        print(f"🤖 Answer: {response}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

