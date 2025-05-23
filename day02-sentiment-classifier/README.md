# Day 02: Sentiment Classifier with LiteLLM and LangChain

Building on Day 01's chat agent, this project extends functionality to create a sentiment analysis chatbot that classifies user messages as Positive, Negative, or Neutral using the DeepSeek-R1 language model.

## What I Learned

- How to extend a basic LLM chat agent to add sentiment analysis capabilities
- Implementing custom methods in a LangChain chat model
- Creating structured data objects with Pydantic for sentiment results
- Using system prompts to guide model behavior for specific tasks
- Enhancing user experience with sentiment feedback and emoji indicators

## Code Structure

- `day01_chat_agent.py`: Original chat implementation (from Day 01)
- `day02_sentiment_classifier.py`: Enhanced implementation with sentiment analysis
- `.env`: Environment file containing API keys (not committed to repository)

## How to Run

1. Set up your environment:
   ```bash
   pip install langchain langchain-core litellm python-dotenv pydantic

Create a .env file with your API key:
HF_API_TOKEN=your_huggingface_token_here

Run the sentiment classifier:
bashpython day02_sentiment_classifier.py

Chat with the AI! Your messages will be analyzed for sentiment and you'll see the classification along with the response. Type 'exit' to quit.

Creating Your .env File

Create a file named .env in the project root directory
Add your Hugging Face API token:
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Get your Hugging Face API token from: https://huggingface.co/settings/tokens
Make sure to keep this file private and add it to your .gitignore

## Technical Details
This sentiment classifier:

- **Extends Day 01's Chat Agent**: Builds upon the LiteLLMChatModel class created on Day 01
- **Adds Sentiment Analysis**: Implements a specialized method to classify text sentiment
- **Uses Prompt Engineering**: Creates a carefully designed system prompt to guide the model's sentiment classification
- **Structured Results**: Uses Pydantic models to create well-defined sentiment result objects
- **Enhanced User Experience**: Provides visual feedback using emoji indicators for different sentiment types
