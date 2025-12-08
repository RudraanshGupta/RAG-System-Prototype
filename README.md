# RAG-System-Prototype

Task Overview
-------------

The objective is to prototype a RAG system that enables efficient document search and LLM-based answering with guardrails. This system is designed to handle a corpus of technical manuals (e.g., cyclone operating manuals).

System Architecture
-------------------

The system follows a standard RAG architecture enhanced with a reranking step for improved answer accuracy:

1.  **PDF Documents**: The input corpus of technical documentation.
    
2.  **Document Loader & Chunker**: PDF files are loaded and split into smaller, semantically coherent chunks.
    
3.  **Embedding Model**: Chunks are converted into numerical vector embeddings using sentence-transformers/all-MiniLM-L6-v2.
    
4.  **FAISS Vector Store**: These embeddings are indexed and stored locally in a FAISS vector database for efficient similarity search.
    
5.  **Retriever**: Upon a user query, FAISS quickly retrieves an initial set of relevant document chunks.
    
6.  **Reranker (Cross-Encoder)**: A cross-encoder/ms-marco-MiniLM-L-6-v2 model then re-scores these chunks and selects the single most relevant one, significantly improving context quality.
    
7.  **LLM (google/flan-t5-large)**: The reranked, most relevant chunk is passed to the locally-run google/flan-t5-large Language Model along with the user's query.
    
8.  **Final Answer (with Sources)**: The LLM generates a natural language answer based _only_ on the provided context, along with citations to the original document snippets.
    

Setup and Running the Prototype
-------------------------------

Follow these steps to set up and run the RAG system:

### 1\. Prerequisites

*   Python 3.8+
    
*   pip package manager
    

### 2\. Project Structure

`   project_folder/  
├── rag_system_.py  
└── docs/      
└── document_1.pdf      
└── document_2.pdf      
└── (a11 files)   `

### 3\. Install Dependencies

Open terminal or command prompt and run the following commands to install all necessary Python packages:

Bash

`   pip install -U langchain langchain-huggingface langchain-community transformers sentence-transformers faiss-cpu PyMuPDF torch   `

### 4\. Place Documents

Create a folder as rag\_system\_final.py. Place all 11 PDF technical documentation files inside this docs folder.

### 5\. Run the System

Execute the Python script:

Bash

`   python rag_system.py   `

Interaction
-----------

*   The script will first perform document ingestion and indexing (this may take a few minutes, especially the first time as models are downloaded and the FAISS index is built).
    
*   Once "RAG chain is ready" appears, it will be prompted to "Ask a question:".
    
*   Type the question related to the content of your PDF documents and press Enter.
    
*   The system will provide an answer and cite the source document snippets it used.
    
*   Type exit or quit to end the session.
    

Important Notes
---------------

*   **Local LLM**: The system uses google/flan-t5-large, which runs locally on machine. This avoids API costs and quotas but requires sufficient RAM and a GPU for faster inference.
    
*   **Vector Store**: The FAISS index (faiss\_index folder) is saved locally. If you add new documents or want to re-index, delete the faiss\_index folder before running the script again.
    
*   **Accuracy**: The reranker significantly improves answer accuracy by ensuring the LLM receives the most relevant context.
