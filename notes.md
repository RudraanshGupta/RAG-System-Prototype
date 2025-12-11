# Design Notes for RAG System Prototype

This document outlines the key design decisions, retrieval strategy, guardrails, and scaling considerations for the Retrieval-Augmented Generation (RAG) system prototype.

## 1. Design Trade-offs & Architecture Choices

The system prioritizes open-source tools and local execution to minimize costs and avoid API dependencies, aligning with the project's requirements for free and open-source components.

**Key Components:**
* **Document Ingestion & Preprocessing:** Handles PDF documents, splits them into manageable chunks.
* **Embeddings & Indexing:** Utilizes `HuggingFaceEmbeddings` with FAISS for efficient similarity search.
* **Retrieval Layer:** Employs a `ContextualCompressionRetriever` with a `CrossEncoderReranker` for enhanced relevance.
* **LLM Layer:** Leverages a local `google/flan-t5-large` model for answer generation via `HuggingFacePipeline`.

**Trade-offs:**
* **Local LLM (Flan-T5 Large):** Provides cost-free operation and data privacy. However, it requires significant local compute resources (RAM/VRAM) and is slower than cloud-hosted, optimized APIs (e.g., OpenAI). The choice of `flan-t5-large` balances quality with reasonable local execution given sufficient resources.
* **FAISS:** Excellent for fast vector search on local machines, suitable for this prototype's document scale. Not inherently distributed for massive scale without additional orchestration.
* **`map_reduce` Chain Type vs. `stuff`:** Initially `map_reduce` was considered for larger document context. However, with the addition of a highly effective `CrossEncoderReranker` to select the *single most relevant chunk* (`top_n=1`), the `stuff` chain type was reinstated. This provides a direct and efficient way to pass the precisely relevant context to the LLM, reducing processing overhead compared to `map_reduce` while avoiding token limits for a single, well-chosen document chunk.


## 2. Retrieval Strategy

The retrieval strategy focuses on maximizing relevance and faithfulness to the source documents.

* **Document Chunking:**
    * **Approach:** `RecursiveCharacterTextSplitter` is used. This splitter attempts to maintain semantically coherent chunks by splitting first by larger units (paragraphs, lines) before individual characters.
    * **Size:** `chunk_size=1000` tokens with `chunk_overlap=200`. This size is chosen to capture sufficient context within each chunk while avoiding excessive length that could dilute focus or exceed embedding model limits. Overlap helps preserve context across split boundaries.
* **Embedding Model:**
    * **Choice:** `sentence-transformers/all-MiniLM-L6-v2`. This model is chosen for its balance of performance, small size, and efficiency. It produces high-quality sentence embeddings suitable for semantic search.
* **Retrieval Method:**
    * **Core Method:** Dense vector search using **FAISS** (Facebook AI Similarity Search) for its speed and efficiency in approximate nearest neighbor search.
    * **Relevance Enhancement (Reranking):** A `ContextualCompressionRetriever` with a `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is employed. The FAISS retriever initially fetches `k=10` candidate documents. The reranker then re-scores these candidates based on query relevance and selects the `top_n=1` most relevant document to pass to the LLM. This significantly boosts the quality of the context provided to the LLM, improving answer accuracy and faithfulness.


## 3. Guardrails & Failure Modes

Guardrails are crucial for robust RAG systems.

* **No Relevant Answers (Graceful Fallback):**
    * **Mechanism:** If the retriever finds no relevant documents (e.g., empty corpus or highly out-of-scope query), the LLM will typically generate a response indicating it cannot find the answer in its knowledge base, rather than hallucinating. The `return_source_documents=True` also allows the user to see if no meaningful sources were retrieved.
    * **Improvement:** For a production system, a more explicit "no answer found" prompt engineering could be implemented, or a confidence score could be used to trigger a predefined fallback message.
* **Hallucinations (Enforcing Source Citations):**
    * **Mechanism:** The `stuff` chain type with a single, highly-relevant document from the reranker, combined with the LLM's instruction to answer *only* from provided context, naturally reduces hallucination. `return_source_documents=True` directly exposes the retrieved text, allowing users to verify the answer against the source snippets.
    * **Improvement:** More advanced prompt engineering ("Answer based *only* on the following context, if the answer is not in the context, state 'I cannot answer this question based on the provided documents.'") could further enforce this.
* **Sensitive Queries:**
    * **Mechanism:** As this prototype uses an open-source, locally run LLM, it inherits the base safety characteristics of the `flan-t5` model. No explicit sensitive query blocking is implemented.
    * **Improvement:** In a production environment, an upstream content moderation API or a dedicated LLM-based classifier would be integrated before the RAG pipeline to filter or flag sensitive inputs.
* **Monitoring Metrics (Conceptual):**
    * **Retrieval Quality:** `Precision@k` (e.g., P@1, P@3) and `Recall@k` would measure how often the most relevant documents are among the top-k retrieved.
    * **Response Faithfulness:** Manual human evaluation or LLM-as-a-judge techniques would be used to assess if answers are grounded *only* in the provided source documents.
    * **Response Relevance:** Evaluating if the answer directly addresses the user's question.


## 4. Scalability Considerations

The design accounts for future scaling, primarily by choosing modular components.

* **10x Increase in Documents (50+ PDFs to 500+):**
    * **Embedding/Indexing:** FAISS can handle a significant increase in documents on a single machine (millions of vectors) if sufficient RAM is available. For hundreds of thousands to millions, a distributed vector database (e.g., Chroma, Weaviate, Pinecone, Milvus, Qdrant) would replace FAISS, or FAISS could be distributed across multiple nodes. The current `create_vector_store` function would be adapted to push embeddings to the chosen distributed store.
    * **Chunking:** Parallelizing the document loading and chunking process would speed up ingestion.
* **100+ Concurrent Users:**
    * **LLM Scaling:** The current local `flan-t5-large` is not suitable for 100+ concurrent users due to latency and resource contention. This would require:
        * **GPU Acceleration:** Deploying the LLM on cloud GPUs (e.g., AWS EC2, GCP A100 instances).
        * **Model Serving Frameworks:** Using optimized serving frameworks like NVIDIA Triton Inference Server, vLLM, or Hugging Face Inference Endpoints.
        * **Containerization & Orchestration:** Packaging the RAG application in Docker containers and deploying it on Kubernetes (e.g., EKS, GKE) for auto-scaling and load balancing.
    * **Vector Database Scaling:** If a distributed vector database is adopted, it would handle concurrent retrieval requests.
* **Cloud Deployment under Cost Constraints:**
    * **Serverless/GPU-Efficient Scaling:**
        * **Serverless LLM:** Utilize services like Hugging Face Inference Endpoints or AWS SageMaker Serverless Inference, which scale down to zero when not in use.
        * **Cost-Optimized GPUs:** Use cloud instances with smaller, more cost-effective GPUs (e.g., NVIDIA T4, L4) or spot instances for batch processing.
        * **Lambda/Cloud Functions:** For the retrieval and preprocessing components, serverless functions could handle document ingestion and query routing, scaling on demand.
        * **Quantized Models:** Using 4-bit or 8-bit quantized versions of the LLM can drastically reduce VRAM requirements and cost while maintaining reasonable performance.
