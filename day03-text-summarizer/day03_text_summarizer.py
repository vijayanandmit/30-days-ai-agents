from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List
from pydantic import Field
import litellm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

token = os.getenv("HF_API_TOKEN")
if not token:
    print("HF_API_TOKEN not found in environment!")
    exit()

os.environ["HF_TOKEN"] = token

class LiteLLMChatModel(BaseChatModel):
    model_name: str = Field(default="huggingface/together/deepseek-ai/DeepSeek-R1")

    def _llm_type(self) -> str:
        return "custom_litellm"

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        litellm_messages = []
        for message in messages:
            role = "user"
            if isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            litellm_messages.append({"role": role, "content": message.content})

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=litellm_messages
            )
            content = response.choices[0].message.content
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            print(f"Error during completion: {e}")
            exit()

# Summarizer Chatbot
chat = LiteLLMChatModel()

print("📄 Text Summarizer - Paste your long text below. Type 'exit' to quit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("📄 Goodbye!")
        break

    messages = [
        SystemMessage(content="You are a helpful summarizer. Given a long input text, summarize it in a few sentences."),
        HumanMessage(content=user_input)
    ]

    result = chat.invoke(messages)
    print(f"\n📄 Summary:\n{result.content}")
