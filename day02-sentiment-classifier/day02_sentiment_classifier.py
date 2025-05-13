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
        # Convert LangChain messages to LiteLLM format
        litellm_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"
            litellm_messages.append({"role": role, "content": message.content})

        # Call LiteLLM
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=litellm_messages
            )
            
            # Debug the response structure
            print(f"LiteLLM response type: {type(response)}")
            
            # Extract content from the LiteLLM response
            # LiteLLM returns a response object with 'choices' field containing messages
            content = ""
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            
            # Create an AI message with the content
            ai_message = AIMessage(content=content)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])  # Single list of generations
            
        except litellm.exceptions.AuthenticationError as e:
            print(f"Authentication error: {e}")
            exit()
        except Exception as e:
            print(f"Error during completion: {e}")
            print(f"Response structure: {response if 'response' in locals() else 'No response'}")
            exit()


# Now you can use LangChain's chat flow to interact with LiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

chat = LiteLLMChatModel()

print("🤖 Hello! I am your AI agent. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Bye!")
        break

    # Prepare chat messages (including system message once)
    messages = [
        SystemMessage(content="You are a sentiment analysis model. Given a user's message, classify it as Positive, Negative, or Neutral."),  # System message
        HumanMessage(content=user_input)  # User message
    ]

    # Get response from LiteLLM using LangChain flow
    result = chat.invoke(messages)

    # Print the AI's response
    print(f"🤖 {result.content}")  # LangChain's invoke() returns AIMessage
