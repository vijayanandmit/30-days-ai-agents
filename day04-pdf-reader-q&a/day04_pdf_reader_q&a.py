from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
from pydantic import Field
from typing import List
import litellm
import os

# Load environment variables
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


# Step 1: Load PDF
pdf_path = "./mastering-ai-agents-galileo.pdf"  
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Step 2: Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

# Step 3: Embed and store in FAISS
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# Step 4: Ask questions using LLM + Retriever
chat = LiteLLMChatModel()

print("📄 PDF Q&A - Ask me anything from the document. Type 'exit' to quit.")
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break

    # Step 5: Retrieve relevant docs
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in relevant_docs[:3]])

    # Step 6: Send to LLM
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the context below to answer the question."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    result = chat.invoke(messages)
    print(f"\n🤖 Answer:\n{result.content}")
