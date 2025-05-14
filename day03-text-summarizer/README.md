# Day 03: Text Summarizer with LiteLLM and LangChain

Building on Day 02's sentiment classifier, this project adds a **text summarization** capability. It allows users to input a long paragraph and receive a short summary, powered by the DeepSeek-R1 language model through LiteLLM and LangChain.

## What I Learned

- How to build a custom text summarizer using LangChain and LiteLLM
- How to design system prompts to instruct the model to perform summarization
- Integrating Hugging Face-hosted models with LangChain via LiteLLM
- Reusing and adapting a base model class to serve multiple use cases
- Simplifying user interaction for summarization tasks in a command-line interface

## Code Structure

- `day01_chat_agent.py`: Original chat implementation (from Day 01)
- `day02_sentiment_classifier.py`: Sentiment classifier using custom system prompts
- `day03_text_summarizer.py`: New summarizer implementation
- `.env`: Environment file containing API keys (not committed to repository)

## How to Run

1. Set up your environment:
   ```bash
   pip install langchain langchain-core litellm python-dotenv pydantic

Create a .env file with your API key:
HF_API_TOKEN=your_huggingface_token_here

Run the sentiment classifier:
python day03_text_summarizer.py

Paste any long paragraph and receive a short summary instantly!

Creating Your .env File

Create a file named .env in the project root directory
Add your Hugging Face API token:
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Get your Hugging Face API token from: https://huggingface.co/settings/tokens
Make sure to keep this file private and add it to your .gitignore

## Technical Details
## Technical Details

This summarizer:

- **Builds on Day 01's Model Class**: Reuses the `LiteLLMChatModel` class
- **Implements a Summarization Task**: Uses a system prompt to instruct the model to summarize long input texts
- **Uses Prompt Engineering**: Includes a clear instruction like  
  `"You are a helpful summarizer. Summarize the input in a few sentences."`
- **Command-Line Based**: Interactive input/output interface in the terminal
- **Simple & Extensible**: Easily adaptable for summarizing documents, emails, articles, etc.
