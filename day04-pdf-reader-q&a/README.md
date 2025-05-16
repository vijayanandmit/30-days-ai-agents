# Day 04: PDF Reader Q&A with LiteLLM, FAISS, and LangChain

Building on earlier projects, this challenge adds a **PDF question-answering agent**. It lets users ask questions about a PDF document and get answers using semantic search and a language model (DeepSeek-R1) via LiteLLM.

## What I Learned

- How to extract text from PDFs using LangChain’s `PyPDFLoader`
- How to chunk and embed text using `sentence-transformers` for semantic search
- How to use `FAISS` to retrieve relevant text chunks from a vector store
- How to send user questions and document context to DeepSeek using a custom LangChain-compatible model
- How to chain document indexing + search + answer generation

## Code Structure

- `day04_pdf_reader_q&a.py`: Main script for loading PDF, creating vector store, and answering user queries
- `LiteLLMChatModel`: Custom class that wraps DeepSeek via `litellm` for LangChain compatibility
- `.env`: Contains your Hugging Face API key

## How to Run

1. **Install dependencies**:
   ```bash
   pip install langchain langchain-core langchain-community langchain-huggingface
   pip install sentence-transformers faiss-cpu pypdf litellm python-dotenv
   ```

2. **Prepare your `.env` file**:
   Create a file named `.env` and add:
   ```env
   HF_API_TOKEN=your_huggingface_token_here
   ```

3. **Add your PDF**:
   Place your PDF file in the project folder and set the file name in the script:
   ```python
   pdf_path = "your-pdf-file.pdf"
   ```

4. **Run the program**:
   ```bash
   python day04_pdf_reader_q&a.py
   ```

5. **Ask questions interactively!**
   Example:
   ```
   You: What is the main topic of the document?
   ```

## Technical Details

This Q&A agent:

- **Uses `PyPDFLoader`** to load and split PDF into chunks
- **Uses `sentence-transformers/all-MiniLM-L6-v2`** to embed both text and user queries
- **Stores chunks in a FAISS vector store** for efficient semantic search
- **Selects top-matching chunks** and feeds them into a prompt to DeepSeek
- **DeepSeek LLM (via LiteLLM)** generates an answer based on context

> Note: `HuggingFaceEmbeddings` should now be imported from `langchain_huggingface` due to deprecation in LangChain 0.2.2.

## Creating Your `.env` File

1. Create a file named `.env` in the project root directory.
2. Add your Hugging Face API token:
   ```env
   HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

Get your Hugging Face API token here: https://huggingface.co/settings/tokens  
Keep this file private and do not commit it to version control.