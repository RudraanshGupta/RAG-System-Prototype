import os
import torch
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

DOCS_PATH = "docs/"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"


#Document Ingestion and Processing 
def create_vector_store():
    print("--- Starting document ingestion and indexing ---")
    print(f"Loading documents from {DOCS_PATH}...")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print("No documents found. Please ensure your 11 PDF files are in the 'docs/' directory.")
        return

    print(f"Loaded {len(documents)} document(s).")
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print(f"Creating embeddings with '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    print(f"Creating and saving FAISS vector store to '{VECTOR_STORE_PATH}'...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print("--- Indexing complete ---")
    
    # 3. LLM and RAG Chain Setup
def setup_qa_chain():
    """Initializes the RAG chain with a reranker for improved accuracy."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    print("Loading vector store...")
    db = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Initializing the reranker...")
    reranker_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

    #FIX: Change top_n from 3 to 1 to ensure the context fits the model
    compressor = CrossEncoderReranker(model=reranker_model, top_n=1)

    base_retriever = db.as_retriever(search_kwargs={"k": 10})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    print(f"Loading local LLM '{LLM_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0.1,
        top_p=0.95,
        device=0 if torch.cuda.is_available() else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True
    )

    print("--- RAG chain is ready ---")
    return qa_chain
