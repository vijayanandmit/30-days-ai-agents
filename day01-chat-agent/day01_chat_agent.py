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


def resolve_provider_config() -> dict:
    """Resolve LiteLLM config for either Hugging Face (default) or Moltbook."""
    provider = os.getenv("LLM_PROVIDER", "huggingface").strip().lower()

    if provider == "moltbook":
        api_key = os.getenv("MOLTBOOK_API_KEY")
        api_base = os.getenv("MOLTBOOK_API_BASE")
        model = os.getenv("MOLTBOOK_MODEL", "openai/gpt-4o-mini")

        if not api_key:
            raise ValueError("MOLTBOOK_API_KEY not found in environment!")
        if not api_base:
            raise ValueError("MOLTBOOK_API_BASE not found in environment!")

        return {
            "provider": "moltbook",
            "model": model,
            "completion_kwargs": {
                "api_key": api_key,
                "api_base": api_base,
            },
        }

    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN not found in environment!")

    os.environ["HF_TOKEN"] = token

    return {
        "provider": "huggingface",
        "model": os.getenv("HF_MODEL", "huggingface/together/deepseek-ai/DeepSeek-R1"),
        "completion_kwargs": {},
    }


CONFIG = resolve_provider_config()


class LiteLLMChatModel(BaseChatModel):
    model_name: str = Field(default=CONFIG["model"])

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
                messages=litellm_messages,
                **CONFIG["completion_kwargs"],
            )

            # Extract content from the LiteLLM response
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content

            # Create an AI message with the content
            ai_message = AIMessage(content=content)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

        except litellm.exceptions.AuthenticationError as e:
            print(f"Authentication error: {e}")
            exit()
        except Exception as e:
            print(f"Error during completion: {e}")
            print(f"Response structure: {response if 'response' in locals() else 'No response'}")
            exit()


# Now you can use LangChain's chat flow to interact with LiteLLM
chat = LiteLLMChatModel()

print(f"🤖 Hello! Provider: {CONFIG['provider']}. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Bye!")
        break

    # Prepare chat messages (including system message once)
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=user_input),
    ]

    # Get response from LiteLLM using LangChain flow
    result = chat.invoke(messages)

    # Print the AI's response
    print(f"🤖 {result.content}")
