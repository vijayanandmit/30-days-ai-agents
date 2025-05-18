# Day 06 - Simple RAG Bot (FAISS + LangChain)

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using HuggingFace embeddings, FAISS for vector storage, and LangChain with a custom LLM wrapper via LiteLLM.

## 🔧 Features

- Vector search powered by FAISS
- HuggingFace embeddings using `all-MiniLM-L6-v2`
- RAG architecture using LangChain retriever + custom DeepSeek LLM
- CLI bot that answers questions from indexed documents

## 📁 Folder Structure

```
day06-rag-bot/
├── day06_rag_bot.py
├── faiss_index/
├── .env
```

## 🧠 How It Works

1. Load a pre-built FAISS vector index from disk
2. Use HuggingFace embeddings for similarity search
3. Retrieve top matching chunks based on the user query
4. Feed the context to a DeepSeek model (via LiteLLM) to generate an answer

## 🚀 Usage

```bash
python day06_rag_bot.py
```

Type a question based on the indexed document content. Type `exit` to quit.

## 🔐 Environment Variables

Ensure you have the following in your `.env` file:

```
HF_API_TOKEN=your_huggingface_token
```

## 🛠 Requirements

- Python 3.10+
- `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface`
- `faiss-cpu`
- `litellm`
- `python-dotenv`

Install with:

```bash
pip install -r requirements.txt
```

## ✅ Note

To remove LangChain deprecation warnings, `retriever.get_relevant_documents(query)` is replaced with `retriever.invoke(query)`