
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field
import litellm
import os
import pickle

# Load HF token
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_API_TOKEN")
if not token:
    print("HF_API_TOKEN not found in .env!")
    exit()
os.environ["HF_TOKEN"] = token

# Custom LLM Wrapper using DeepSeek
class LiteLLMChatModel(BaseChatModel):
    model_name: str = Field(default="huggingface/together/deepseek-ai/DeepSeek-R1")

    def _llm_type(self) -> str:
        return "custom_litellm"

    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
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

# Load FAISS index and docs
faiss_folder = "faiss_index"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(faiss_folder, embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Initialize LLM
chat = LiteLLMChatModel()

# RAG Loop
print("🧠 Simple RAG Bot - Ask me anything! Type 'exit' to quit.")
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break

    ##docs = retriever.get_relevant_documents(query)
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs[:3]])

    messages = [
        SystemMessage(content="You are a helpful assistant. Use the context below to answer the question."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = chat.invoke(messages)
    print(f"\n🤖 Answer:\n{response.content}")
