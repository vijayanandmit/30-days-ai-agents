from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import pickle

# --------------------------
# Step 1: Load Environment
# --------------------------
load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")

if not hf_token:
    raise ValueError("HF_API_TOKEN not found in .env file. Please create a .env file with HF_API_TOKEN=<your_token>")

os.environ["HF_TOKEN"] = hf_token

# --------------------------
# Step 2: Load PDF File
# --------------------------
pdf_path = "./Databricks-Big-Book-Of-GenAI-FINAL.pdf"  # Change to your PDF file path
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

# --------------------------
# Step 3: Split into Chunks
# --------------------------
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

print(f"✅ Loaded and split into {len(docs)} chunks.")

# --------------------------
# Step 4: Initialize Embeddings
# --------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# Step 5: Create FAISS Vector Store
# --------------------------
vectorstore = FAISS.from_documents(docs, embedding_model)

# --------------------------
# Step 6: Save Vector Store
# --------------------------
index_path = "faiss_index"
vectorstore.save_local(index_path)

# Save documents separately for later inspection (optional)
with open(os.path.join(index_path, "documents.pkl"), "wb") as f:
    pickle.dump(docs, f)

print("🎉 Embeddings generated and FAISS index saved to 'faiss_index'")
