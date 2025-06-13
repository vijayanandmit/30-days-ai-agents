# Day 05: Embeddings Generator

This day focuses on creating a FAISS vector store from a PDF document. The script splits a PDF into chunks, generates embeddings using a HuggingFace model and saves them for later retrieval.

## What I Learned

- How to extract text from PDFs using LangChain’s `PyPDFLoader`
- How to chunk and embed text using `sentence-transformers` for semantic search
- How to use `FAISS` to retrieve relevant text chunks from a vector store
- How to generate embeddings for document chunks
- How to store embeddings for fast retrieval later

## Code Structure

- `day05-embeddings-generator.py`: Reads the PDF and builds the FAISS index
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
   python day05-embeddings-generator.py
   ```

## Technical Details

This script:

- **Uses `PyPDFLoader`** to load and split the PDF into manageable chunks
- **Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`**
- **Stores the embeddings in a FAISS vector store** for later retrieval

> Note: `HuggingFaceEmbeddings` should now be imported from `langchain_huggingface` due to deprecation in LangChain 0.2.2.

## Creating Your `.env` File

1. Create a file named `.env` in the project root directory.
2. Add your Hugging Face API token:
   ```env
   HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

Get your Hugging Face API token here: https://huggingface.co/settings/tokens  
Keep this file private and do not commit it to version control.