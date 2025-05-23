
# Day 01: Chat Agent with LiteLLM and LangChain

Built a simple chat agent that connects LangChain with LiteLLM to access large language models like DeepSeek-R1.

## What I Learned

- How to use LangChain's `BaseChatModel` to create custom model integrations
- How to connect to AI models through LiteLLM
- Properly handling message formats between different APIs
- Troubleshooting common integration errors like response parsing issues

## Code Structure

- `day01_chat_agent.py`: Main implementation of a chat loop using LangChain and LiteLLM
- `.env`: Environment file containing API keys (not committed to repository)

## How to Run

1. Set up your environment:
   ```bash
   pip install langchain langchain-core litellm python-dotenv
   ```

2. Create a `.env` file with your API key:
   ```
   HF_API_TOKEN=your_huggingface_token_here
   ```

3. Run the chat agent:
   ```bash
   python day01_chat_agent.py
   ```

4. Chat with the AI! Type 'exit' to quit.

## Creating Your .env File

1. Create a file named `.env` in the project root directory
2. Add your Hugging Face API token:
   ```
   HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Get your Hugging Face API token from: https://huggingface.co/settings/tokens
4. Make sure to keep this file private and add it to your `.gitignore`

## Technical Details

This integration:
- Creates a custom LangChain chat model class
- Converts messages between LangChain and LiteLLM formats
- Properly extracts AI responses from the LiteLLM response object
- Formats the output for the user in a simple chat interface

